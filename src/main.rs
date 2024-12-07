use crabgrab::{
    capturable_content::{CapturableContent, CapturableContentFilter},
    capture_stream::{CaptureAccessToken, CaptureConfig, CapturePixelFormat, CaptureStream},
    feature::bitmap::{FrameBitmap, FrameBitmapYCbCr, VideoFrameBitmap},
    prelude::{BitmapDataChroma, BitmapDataLuma},
};

use ffmpeg_next::{
    self as ffmpeg, codec, encoder, log,
    format::{self, context::Output},
    frame, Dictionary, Packet, Rational,
};

use tokio::{
    io::{AsyncBufReadExt, BufReader},
    sync::mpsc::{self, Receiver},
    task::JoinHandle,
    time::{self, Duration},
};

use std::{
    io::{self, Write},
    slice,
    sync::{atomic::{AtomicBool, Ordering}, Arc, Mutex},
};

use chrono::Local;

const DEFAULT_X264_OPTS: &str = "preset=medium";

struct CustomEncoder {
    encoder: encoder::Video,
    input_time_base: Rational,
    frame_count: usize,
}

impl CustomEncoder {
    fn new(
        octx: &mut Output,
        x264_opts: Dictionary,
        width: f64,
        height: f64,
        framerate: i32,
    ) -> Result<Self, ffmpeg::Error> {
        let global_header = octx.format().flags().contains(format::Flags::GLOBAL_HEADER);

        let codec = encoder::find(codec::Id::H264);
        let mut ost = octx.add_stream(codec)?;

        let mut encoder =
            codec::context::Context::new_with_codec(codec.ok_or(ffmpeg::Error::InvalidData)?)
                .encoder()
                .video()?;
        ost.set_parameters(&encoder);
        encoder.set_width(width as u32);
        encoder.set_height(height as u32);
        encoder.set_format(format::Pixel::YUV420P);
        encoder.set_time_base(Rational::new(1, framerate * 100));

        if global_header {
            encoder.set_flags(codec::Flags::GLOBAL_HEADER);
        }

        let opened_encoder = encoder
            .open_with(x264_opts)
            .expect("error opening x264 with supplied settings");
        ost.set_parameters(&opened_encoder);

        Ok(Self {
            encoder: opened_encoder,
            input_time_base: Rational::new(1, framerate * 100),
            frame_count: 0,
        })
    }

    fn fill_ffmpeg_frame<Y, UV>(
        &mut self,
        luma_data: &Y,    // Y (Luma) plane data
        chroma_data: &UV, // UV (Chroma) plane data
        width: usize,
        height: usize,
    ) -> Result<frame::Video, Box<dyn std::error::Error>>
    where
        Y: BitmapDataLuma,
        UV: BitmapDataChroma,
    {
        let luma = luma_data.as_ref();
        let chroma = chroma_data.as_ref();

        // Validate input data size
        let luma_size = width * height; // Y plane size
        let chroma_size = (width / 2) * (height / 2); // Each of U and V (for YUV420)

        if luma.len() != luma_size || chroma.len() != chroma_size {
            return Err("Input data size does not match dimensions".into());
        }

        // Create frame
        let mut frame = frame::Video::new(
            format::Pixel::YUV420P,
            width.try_into()?,
            height.try_into()?,
        );

        unsafe {
            let data = [
                slice::from_raw_parts_mut(frame.data_mut(0).as_mut_ptr(), frame.data_mut(0).len()),
                slice::from_raw_parts_mut(frame.data_mut(1).as_mut_ptr(), frame.data_mut(1).len()),
                slice::from_raw_parts_mut(frame.data_mut(2).as_mut_ptr(), frame.data_mut(2).len()),
            ];
            let linesize: [usize; 3] = [frame.stride(0), frame.stride(1), frame.stride(2)];

            // Fill Luma (Y) plane
            let y_stride = linesize[0] as usize;
            for row in 0..height {
                let src_start = row * width;
                let dest_start = row * y_stride;
                data[0][dest_start..dest_start + width]
                    .copy_from_slice(&luma[src_start..src_start + width]);
            }

            // Fill Chroma (CbCr) planes (interleaved as UVUV...)
            let uv_stride = linesize[1] as usize;
            for row in 0..(height / 2) {
                let src_start = row * (width / 2);
                let dest_start = row * uv_stride;
                for col in 0..(width / 2) {
                    if dest_start + col >= data[1].len() || dest_start + col >= data[2].len() {
                        panic!("Index out of bounds: dest_start + col exceeds buffer size");
                    }

                    let cb = chroma[src_start + col][0];
                    let cr = chroma[src_start + col][1];

                    data[1][dest_start + col] = cb;
                    data[2][dest_start + col] = cr;
                }
            }
        }

        Ok(frame)
    }

    fn receive_and_process_decoded_frames(
        &mut self,
        octx: &mut Output,
        bitmap: FrameBitmapYCbCr<Box<[u8]>, Box<[[u8; 2]]>>,
        ost_time_base: Rational,
    ) {
        let mut input_frame = self
            .fill_ffmpeg_frame(
                &bitmap.luma_data,
                &bitmap.chroma_data,
                bitmap.luma_width,
                bitmap.luma_height,
            )
            .unwrap();

        let timestamp = (self.frame_count * 100) as i64;
        input_frame.set_pts(Some(timestamp));

        self.frame_count += 1;

        self.send_frame_to_encoder(&input_frame);
        self.receive_and_process_encoded_packets(octx, ost_time_base);
    }

    fn send_frame_to_encoder(&mut self, frame: &frame::Video) {
        self.encoder.send_frame(frame).unwrap();
    }

    fn send_eof_to_encoder(&mut self) {
        self.encoder.send_eof().unwrap();
    }

    fn receive_and_process_encoded_packets(&mut self, octx: &mut Output, ost_time_base: Rational) {
        let mut encoded = Packet::empty();
        while self.encoder.receive_packet(&mut encoded).is_ok() {
            encoded.rescale_ts(self.input_time_base, ost_time_base);
            encoded.write_interleaved(octx).unwrap();
        }
    }
}

fn parse_opts<'a>(s: String) -> Option<Dictionary<'a>> {
    let mut dict = Dictionary::new();
    for keyval in s.split_terminator(',') {
        let tokens: Vec<&str> = keyval.split('=').collect();
        match tokens[..] {
            [key, val] => dict.set(key, val),
            _ => return None,
        }
    }
    Some(dict)
}

async fn capture_frame(
    token: CaptureAccessToken,
    config: CaptureConfig,
    encoder: Arc<Mutex<CustomEncoder>>,
    octx: Arc<Mutex<Output>>,
    ost_time_base: Rational,
) {
    match crabgrab::feature::screenshot::take_screenshot(token, config).await {
        Ok(frame) => {
            if let Ok(FrameBitmap::YCbCr(bitmap)) = frame.get_bitmap() {
                encoder.lock().unwrap().receive_and_process_decoded_frames(
                    &mut octx.lock().unwrap(),
                    bitmap,
                    ost_time_base,
                );
            }
        }
        Err(_) => println!("screenshot failed!"),
    }
}

async fn record_task<'a>(
    rx: Arc<tokio::sync::Mutex<Receiver<&'a str>>>,
    encoder: Arc<Mutex<CustomEncoder>>,
    octx: Arc<Mutex<Output>>,
    ost_time_base: Rational,
    output_path: &'a str,
    interval: Duration,
    config: CaptureConfig,
    token: CaptureAccessToken,
) {
    let mut tasks: Vec<JoinHandle<()>> = Vec::new();
    let mut ticker = time::interval(interval);

    println!("Recording entire screen");

    loop {
        if let Ok(command) = rx.lock().await.try_recv() {
            match command {
                "stop" => {
                    println!("Stopping recording");

                    for task in tasks.drain(..) {
                        task.await.unwrap();
                    }

                    encoder.lock().unwrap().send_eof_to_encoder();
                    encoder.lock().unwrap().receive_and_process_encoded_packets(
                        &mut octx.lock().unwrap(),
                        ost_time_base,
                    );

                    octx.lock().unwrap().write_trailer().unwrap();

                    println!("Recording saved to {}", output_path);
                }
                "break" => {
                    println!("Breaking");
                    break;
                }
                _ => {
                    println!("Unknown command: {}. Try 'help'", command);
                }
            }
        }

        ticker.tick().await;
        println!("Recording frame");

        let copied_config = config.clone();
        let copied_cencoder = Arc::clone(&encoder);
        let copied_octx = Arc::clone(&octx);

        let handle = tokio::spawn(async move {
            capture_frame(
                token,
                copied_config,
                copied_cencoder,
                copied_octx,
                ost_time_base,
            )
            .await;
        });

        tasks.push(handle);
    }
}

#[tokio::main]
async fn main() {
    ffmpeg::init().unwrap();
    log::set_level(log::Level::Error);

    let token = match CaptureStream::test_access(false) {
        Some(token) => token,
        None => CaptureStream::request_access(false)
            .await
            .expect("Expected capture access"),
    };
    let filter = CapturableContentFilter::DISPLAYS;
    let content = CapturableContent::new(filter).await.unwrap();
    let config =
        CaptureConfig::with_display(content.displays().next().unwrap(), CapturePixelFormat::V420);

    let now = Local::now();
    let time_string = now.format("%Y-%m-%d_%H-%M-%S").to_string();
    let output_path = format!("Screen_recording_{}.mp4", time_string);

    let octx = Arc::new(Mutex::new(format::output(&output_path).unwrap()));

    let x264_opts = parse_opts(DEFAULT_X264_OPTS.to_string()).expect("Failed to parse x264_opts");

    let screen = content.displays().next().unwrap();
    let width = screen.rect().size.width;
    let height = screen.rect().size.height;

    let frame_rate = 30;

    let encoder = Arc::new(Mutex::new(
        CustomEncoder::new(
            &mut octx.lock().unwrap(),
            x264_opts.to_owned(),
            width,
            height,
            frame_rate,
        )
        .unwrap(),
    ));

    format::context::output::dump(&mut octx.lock().unwrap(), 0, Some(&output_path));
    octx.lock().unwrap().write_header().unwrap();

    let ost_time_base = octx.lock().unwrap().stream(0).unwrap().time_base();

    let interval = Duration::from_secs_f64(1.0 / frame_rate as f64);

    let (tx, mut rx) = mpsc::channel(32);

    let is_recording = Arc::new(AtomicBool::new(false));
    let is_recording_clone = Arc::clone(&is_recording);

    let input_thread = tokio::spawn(async move {
        let stdin = tokio::io::stdin();
        let mut stdin = BufReader::new(stdin);
        let mut line = String::new();

        loop {
            line.clear();
            print!("> ");
            io::stdout().flush().unwrap();

            match stdin.read_line(&mut line).await {
                Ok(_) => {
                    let input = line.trim();
                    match input {
                        "help" => {
                            println!("Available commands:");
                            println!("s - record entire screen");
                            println!("q - stop recording");
                            println!("exit - exit");
                        }
                        "s" => {
                            println!("Starting recording...");
                            println!("Enter q to stop");
                            if is_recording_clone.load(Ordering::SeqCst) {
                                println!("Already recording");
                                continue;
                            }
                            is_recording_clone.store(true, Ordering::SeqCst);
                            tx.send("start").await.unwrap();
                        }
                        "q" => {
                            if !is_recording_clone.load(Ordering::SeqCst) {
                                println!("Not recording");
                                continue;
                            }
                            println!("Stopping recording...");
                            tx.send("stop").await.unwrap();
                            break;
                        }
                        "exit" => {
                            tx.send("break").await.unwrap();
                            break;
                        }
                        _ => {
                            println!("Unknown command: {}. Try 'help'", input);
                        }
                    }
                }
                Err(e) => {
                    println!("Error reading input: {}", e);
                    break;
                }
            }
        }
    });

    let recording_thread = tokio::spawn(async move {
        let mut is_recording = false;
        let mut tasks: Vec<JoinHandle<()>> = Vec::new();
        let mut ticker = time::interval(interval);

        loop {
            if let Ok(command) = rx.try_recv() {
                match command {
                    "start" => {
                        if !is_recording {
                            is_recording = true;
                        }
                    }
                    "stop" => {
                        for task in tasks.drain(..) {
                            task.await.unwrap();
                        }

                        encoder.lock().unwrap().send_eof_to_encoder();
                        encoder.lock().unwrap().receive_and_process_encoded_packets(
                            &mut octx.lock().unwrap(),
                            ost_time_base,
                        );

                        octx.lock().unwrap().write_trailer().unwrap();

                        println!("Recording saved to {}", output_path);
                        break;
                    }
                    "break" => {
                        break;
                    }
                    _ => {
                        println!("Unknown command: {}. Try 'help'", command);
                    }
                }
            }

            if is_recording {
                ticker.tick().await;

                let copied_config = config.clone();
                let copied_cencoder = Arc::clone(&encoder);
                let copied_octx = Arc::clone(&octx);

                let handle = tokio::spawn(async move {
                    capture_frame(
                        token,
                        copied_config,
                        copied_cencoder,
                        copied_octx,
                        ost_time_base,
                    )
                    .await;
                });

                tasks.push(handle);
            }
        }
    });

    let _ = tokio::join!(input_thread, recording_thread);
}
