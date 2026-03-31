use image::{ImageReader, DynamicImage, GenericImageView, Pixel, Rgb};
use std::io::BufReader;
use std::fs;

enum Digit {
    Zero,
    One,
    Two,
    Three,
    Four,
    Five,
    Six,
    Seven,
    Eight,
    Nine,
}

const MNIST_WIDTH: usize = 28;
const MNIST_HEIGHT: usize = MNIST_WIDTH;

struct MNISTImage {
    data: [u8; MNIST_WIDTH * MNIST_HEIGHT],

}

struct ClassedImage {
    image: MNISTImage,
    class: Digit,
}

impl MNISTImage {
    fn blank() -> MNISTImage {
        MNISTImage {
            data: [0; 28 * 28],
        }
    }

    fn from_file(filename: &str) -> Result<MNISTImage, &'static str> {
        match ImageReader::open(filename) {
            Ok(reader) => Self::from_imagereader(reader),
            Err(_) => Err("Error opening image file."),
        }
    }

    fn from_imagereader(reader: ImageReader<BufReader<fs::File>>) -> Result<MNISTImage, &'static str> {
        match reader.decode() {
            Ok(imagedata) => Self::from_dynamicimage(imagedata),
            Err(msg) => Err("Error decoding image file."),
        }
    }

    fn average_pixel_channels(pix: Rgb<u8>) -> u8 {
        let mut sum: u16 = 0;
        for i in 0..=2 {
            sum += u16::from(pix[i]);
        }

        match u8::try_from(sum / 3) {
            Ok(avg) => avg,
            Err(_) => panic!("Tried to average pixel ({},{},{}), found value greater than 255.", pix[0],pix[1],pix[2]),
        }
    }

    fn from_dynamicimage(image_data: DynamicImage) -> Result<MNISTImage, &'static str> {
        let mut i = 0;
        let mut new_image = MNISTImage::blank();
        for pix in image_data.pixels() {
            new_image.data[i] = Self::average_pixel_channels(pix.2.to_rgb());
            i += 1;
        }
        Ok(new_image)
    }

    fn print(&self) {
        let mut output = String::new();
        let mut i = 0;
        for value in self.data {
            output += &(value.to_string() + " ");
            if value < 10 {
                output += " ";
            }
            if value < 100 {
                output += " ";
            }
            if i % MNIST_WIDTH == MNIST_WIDTH - 1 {
                output += "\n"
            }
            i += 1;
        }
        println!("{}",output)
    }
}

fn main() {
    let img = MNISTImage::from_file("mnist_png/train/0/1.png");
    match img {
        Ok(img) => img.print(),
        Err(msg) => println!("{}",msg),
    }
}
