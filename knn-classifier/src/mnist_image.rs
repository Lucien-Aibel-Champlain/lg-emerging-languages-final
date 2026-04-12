use image::{ImageReader, DynamicImage, GenericImageView, Pixel, Rgb};
use std::io::BufReader;
use std::fs;

const MNIST_WIDTH: usize = 28;
const MNIST_HEIGHT: usize = MNIST_WIDTH;

pub struct MNISTImage {
    data: [u8; MNIST_WIDTH * MNIST_HEIGHT],
}

pub struct ClassedImage {
    pub image: MNISTImage,
    pub class: u8,
}

impl MNISTImage {
    pub fn blank() -> MNISTImage {
        MNISTImage {
            data: [0; 28 * 28],
        }
    }

    pub fn from_file(filename: &str) -> Result<MNISTImage, String> {
        match ImageReader::open(filename) {
            Ok(reader) => Self::from_imagereader(reader),
            Err(_) => Err("Error opening image file.".to_string()),
        }
    }

    fn from_imagereader(reader: ImageReader<BufReader<fs::File>>) -> Result<MNISTImage, String> {
        match reader.decode() {
            Ok(imagedata) => Self::from_dynamicimage(imagedata),
            Err(_) => Err("Error decoding image file.".to_string()),
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

    fn from_dynamicimage(image_data: DynamicImage) -> Result<MNISTImage, String> {
        let mut i = 0;
        let mut new_image = MNISTImage::blank();
        for pix in image_data.pixels() {
            new_image.data[i] = Self::average_pixel_channels(pix.2.to_rgb());
            i += 1;
        }
        Ok(new_image)
    }

    pub fn print(&self) {
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

    pub fn calculate_distance(&self, other: &MNISTImage) -> f64 {
        let accumulator: f64 = self.data.iter().zip(other.data.iter()).map(|(a, b)| ((*a as i16 - *b as i16) as f64).powf(2f64)).sum();
        return accumulator.sqrt()
    }
}