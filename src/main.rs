use crabgrab::{
    capturable_content::{CapturableContent, CapturableContentFilter},
    capture_stream::{CaptureAccessToken, CaptureConfig, CapturePixelFormat, CaptureStream},
    feature::bitmap::{FrameBitmap, FrameBitmapYCbCr, VideoFrameBitmap},
    prelude::{BitmapDataChroma, BitmapDataLuma},
};

use ffmpeg_next::{
    self as ffmpeg, codec, encoder,
    format::{self, context::Output},
    frame, log, Dictionary, Packet, Rational,
};

use tokio::{
    io::{AsyncBufReadExt, BufReader},
    sync::{
        mpsc::{self},
        Mutex,
    },
    task::JoinHandle,
    time::{self, Duration},
};

use std::{
    io::{self, Write},
    slice,
    sync::Arc,
};

use chrono::Local;

const DEFAULT_X264_OPTS: &str = "preset=medium";

struct SharedRecorder {
    input_time_base: Rational,
    output_time_base: Rational,
    encoder: Arc<Mutex<encoder::Video>>,
    output_context: Arc<Mutex<Output>>,
}

impl SharedRecorder {
    async fn receive_and_process_encoded_packets(&self) {
        let mut encoded = Packet::empty();
        while self
            .encoder
            .lock()
            .await
            .receive_packet(&mut encoded)
            .is_ok()
        {
            encoded.rescale_ts(self.input_time_base, self.output_time_base);
            let mut octx_value = self.output_context.lock().await;
            encoded.write_interleaved(&mut octx_value).expect("Failed to write packet");
        }
    }
}

struct Recorder {
    shared: Arc<Mutex<SharedRecorder>>,
    config: CaptureConfig,
    token: CaptureAccessToken,
    output_path: String,
    frame_count: Arc<Mutex<usize>>,
}

impl Recorder {
    async fn new(
        width: f64,
        height: f64,
        frame_rate: i32,
        config: CaptureConfig,
        token: CaptureAccessToken,
    ) -> Result<Self, ffmpeg::Error> {
        let now = Local::now();
        let time_string = now.format("%Y-%m-%d_%H-%M-%S").to_string();
        let output_path = format!("Screen_recording_{}.mp4", time_string);
        // let output_path = String::from("output.mp4");

        let output_context_arc = Arc::new(Mutex::new(format::output(&output_path).expect("Failed to create output context"))) ;
        let mut output_context = output_context_arc.lock().await;

        let codec = encoder::find(codec::Id::H264);

        let mut encoder = codec::context::Context::new_with_codec(
            codec.ok_or(ffmpeg::Error::InvalidData).expect("Codec not found"),
        )
        .encoder()
        .video()
        .expect("Failed to get video encoder");
        encoder.set_width(width as u32);
        encoder.set_height(height as u32);
        encoder.set_format(format::Pixel::YUV420P);
        encoder.set_time_base(Rational::new(1, frame_rate * 100));

        let global_header = output_context
            .format()
            .flags()
            .contains(format::Flags::GLOBAL_HEADER);
        if global_header {
            encoder.set_flags(codec::Flags::GLOBAL_HEADER);
        }

        let mut output_stream = output_context.add_stream(codec)?;
        output_stream.set_parameters(&encoder);

        let x264_opts =
            parse_opts(DEFAULT_X264_OPTS.to_string()).expect("Failed to parse x264_opts");
        let opened_encoder = Arc::new(Mutex::new(
            encoder
                .open_with(x264_opts)
                .expect("error opening x264 with supplied settings"),
        ));
        output_stream.set_parameters(&*opened_encoder.lock().await);

        format::context::output::dump(&output_context, 0, Some(&output_path));
        output_context.write_header().expect("Failed to write header");

        let output_time_base = output_context.stream(0).expect("Failed to get output stream").time_base();

        Ok(Self {
            shared: Arc::new(Mutex::new(SharedRecorder {
                input_time_base: Rational::new(1, frame_rate * 100),
                output_time_base,
                encoder: opened_encoder.clone(),
                output_context: output_context_arc.clone(),
            })),
            config,
            token,
            output_path,
            frame_count: Arc::new(Mutex::new(0)),
        })
    }

    async fn record_frame(&self, tasks: &mut Vec<JoinHandle<()>>) {
        let token = self.token.clone();
        let config = self.config.clone();

        let frame_count_arc = self.frame_count.clone();

        let shared_arc = self.shared.clone();

        let handle = tokio::spawn(async move {
            match crabgrab::feature::screenshot::take_screenshot(token, config).await {
                Ok(frame) => {
                    if let Ok(FrameBitmap::YCbCr(bitmap)) = frame.get_bitmap() {
                        let mut frame = encode_bitmap(bitmap);
                        let mut frame_count_value = frame_count_arc.lock().await;
                        let timestamp = (*frame_count_value * 100) as i64;
                        frame.set_pts(Some(timestamp));

                        *frame_count_value += 1;

                        let shared = shared_arc.lock().await;

                        shared.encoder.lock().await.send_frame(&frame).expect("Failed to send frame");
                        shared.receive_and_process_encoded_packets().await;
                    }
                }
                Err(_) => println!("Screenshot failed!"),
            }
        });

        tasks.push(handle);
    }

    async fn stop_recording(&self) {
        let shared = self.shared.lock().await;
        shared.encoder.lock().await.send_eof().expect("Failed to send EOF");
        shared.receive_and_process_encoded_packets().await;

        shared.output_context.lock().await.write_trailer().expect("Failed to write trailer");

        println!("Recording saved to {}", self.output_path);
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

fn encode_bitmap(bitmap: FrameBitmapYCbCr<Box<[u8]>, Box<[[u8; 2]]>>) -> frame::Video {
    fill_ffmpeg_frame(
        &bitmap.luma_data,
        &bitmap.chroma_data,
        bitmap.luma_width,
        bitmap.luma_height,
    )
    .expect("Failed to fill ffmpeg frame")
}

fn fill_ffmpeg_frame<Y, UV>(
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

fn print_help() {
    println!("Available commands:");
    println!("s - record entire screen");
    println!("q - stop recording");
    println!("exit - exit");
}

enum Command {
    Start,
    Stop,
    Exit,
}

#[tokio::main]
async fn main() {
    ffmpeg::init().expect("Failed to initialize ffmpeg");
    log::set_level(log::Level::Error);

    let token = match CaptureStream::test_access(false) {
        Some(token) => token,
        None => CaptureStream::request_access(false)
            .await
            .expect("Expected capture access"),
    };
    let filter = CapturableContentFilter::DISPLAYS;
    let content = CapturableContent::new(filter).await.expect("Failed to get content");
    let config =
        CaptureConfig::with_display(content.displays().next().expect("Expected at least one display"), CapturePixelFormat::V420);

    let screen = content.displays().next().expect("Expected at least one display");
    let width = screen.rect().size.width;
    let height = screen.rect().size.height;

    let frame_rate = 30;

    let (tx, mut rx) = mpsc::channel(100);

    print_help();

    let input_thread = tokio::spawn(async move {
        let stdin = tokio::io::stdin();
        let mut stdin = BufReader::new(stdin);
        let mut line = String::new();
        let mut is_recording = false;

        loop {
            line.clear();
            print!("> ");
            io::stdout().flush().expect("Failed to flush stdout");

            match stdin.read_line(&mut line).await {
                Ok(_) => {
                    let input = line.trim();
                    match input {
                        "help" => {
                            print_help();
                        }
                        "s" => {
                            if is_recording {
                                println!("Already recording");
                                continue;
                            }
                            println!("Recording started. Enter 'q' to stop");
                            is_recording = true;
                            tx.send(Command::Start).await.expect("Failed to send command start");
                        }
                        "q" => {
                            if !is_recording {
                                println!("Not recording");
                                continue;
                            }
                            println!("Stopping recording... close the app if it takes more than a few seconds");
                            tx.send(Command::Stop).await.expect("Failed to send command stop");
                            break;
                        }
                        "exit" => {
                            tx.send(Command::Exit).await.expect("Failed to send command exit");
                            break;
                        }
                        _ => {
                            println!("Unknown command. Try 'help'");
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

    let mut is_recording = false;
    let mut tasks: Vec<JoinHandle<()>> = Vec::new();

    let interval = Duration::from_secs_f64(1.0 / frame_rate as f64);
    let mut ticker = time::interval(interval);

    let mut recorder: Option<Recorder> = None;

    loop {
        if let Ok(command) = rx.try_recv() {
            match command {
                Command::Start => {
                    if !is_recording {
                        is_recording = true;
                        recorder = Some(
                            Recorder::new(width, height, frame_rate, config.clone(), token)
                                .await
                                .expect("Failed to create recorder"),
                        );
                    }
                }
                Command::Stop => {
                    if let Some(rec) = &recorder {
                        for task in tasks.drain(..) {
                            task.await.expect("Failed to stop recording");
                        }
                        rec.stop_recording().await;
                        break;
                    }
                }
                Command::Exit => {
                    for task in tasks.drain(..) {
                        task.await.expect("Failed to stop recording");
                    }
                    break;
                }
            }
        }

        if is_recording {
            if let Some(rec) = &recorder {
                ticker.tick().await;
                rec.record_frame(&mut tasks).await;
            }
        }
    }

    input_thread.await.expect("Failed to join input thread");
}
