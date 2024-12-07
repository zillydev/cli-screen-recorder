// fn make_scaled_base64_png_from_bitmap<DataBgra: BitmapDataBgra8x4>(
//     bitmap: FrameBitmapBgraUnorm8x4<DataBgra>,
//     max_width: usize,
//     max_height: usize,
// ) -> Vec<u8> {
//     let (mut height, mut width) = (bitmap.width, bitmap.height);
//     if width > max_width {
//         width = max_width;
//         height = ((max_width as f64 / bitmap.width as f64) * bitmap.height as f64).ceil() as usize;
//     };
//
//     if height > max_height {
//         height = max_height;
//         width = ((max_height as f64 / bitmap.height as f64) * bitmap.width as f64).ceil() as usize;
//     };
//
//     let mut write_vec = vec![0u8; 0];
//     {
//         let mut encoder = png::Encoder::new(&mut write_vec, width as u32, height as u32);
//         encoder.set_color(png::ColorType::Rgba);
//         encoder.set_depth(png::BitDepth::Eight);
//         let mut writer = encoder.write_header().unwrap();
//         let mut image_data = vec![0u8; width * height * 4];
//         for y in 0..height {
//             let sample_y = (bitmap.height * y) / height;
//             for x in 0..width {
//                 let sample_x = (bitmap.width * x) / width;
//                 let [b, g, r, a] = bitmap.data.as_ref()[sample_x + sample_y * bitmap.width];
//                 image_data[(x + y * width) * 4 + 0] = r;
//                 image_data[(x + y * width) * 4 + 1] = g;
//                 image_data[(x + y * width) * 4 + 2] = b;
//                 image_data[(x + y * width) * 4 + 3] = a;
//             }
//         }
//         writer.write_image_data(&image_data).unwrap();
//     }
//     write_vec
// }

// fn convert_to_rgb24<DataBgra: BitmapDataBgra8x4>(
//     bitmap: FrameBitmapBgraUnorm8x4<DataBgra>,
//     // scaler: &mut ffmpeg_next::software::scaling::context::Context,
//     pts: i64,
//     // encoder: &mut ffmpeg_next::codec::encoder::video::Video,
//     &mut encoder: ffmpeg_next::codec::encoder::video::Video,
// ) {
//     let rgb_data: Vec<u8> = bitmap
//         .data
//         .as_ref()
//         .iter()
//         .flat_map(|&[b, g, r, _a]| vec![r, g, b]) // Drop alpha channel
//         .collect();
//
//     let mut input_frame = ffmpeg::frame::Video::new(ffmpeg::format::Pixel::RGB24, 1920, 1080);
//     input_frame.data_mut(0).copy_from_slice(&rgb_data);
//     // input_frame.set_linesize(0, width * 3);
//
//     let mut output_frame = ffmpeg::frame::Video::new(ffmpeg::format::Pixel::YUV420P, 1920, 1080);
//     // scaler.run(&input_frame, &mut output_frame).unwrap();
//
//     output_frame.set_pts(Some(pts));
//     encoder.send_frame(&output_frame).unwrap();
// }

// fn convert_to_rgb24<DataBgra: BitmapDataBgra8x4>(
//     bitmap: &mut FrameBitmapBgraUnorm8x4<DataBgra>,
// ) -> Vec<u8> {
//     bitmap
//         .data
//         .as_ref()
//         .iter()
//         .flat_map(|&[b, g, r, _a]| vec![r, g, b]) // Drop alpha channel
//         .collect()
//
//     // bitmap
//     //     .data
//     //     .as_ref()
//     //     .chunks(width) // Iterate over each row (chunks of 'width' pixels)
//     //     .flat_map(|row| {
//     //         // For each row, we will extract the RGB values and add padding at the end of the row
//     //         row.iter()
//     //             .flat_map(|&[r, g, b, _a]| vec![r, g, b]) // Extract RGB (discard alpha)
//     //             .chain(vec![0; padding]) // Add padding (stride) at the end of the row
//     //     })
//     //     .collect()
// }