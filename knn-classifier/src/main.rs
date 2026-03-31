use image::{ImageReader, DynamicImage, GenericImageView, Pixel, Rgb};
use std::io::BufReader;
use std::fs;

const MNIST_WIDTH: usize = 28;
const MNIST_HEIGHT: usize = MNIST_WIDTH;

struct MNISTImage {
    data: [u8; MNIST_WIDTH * MNIST_HEIGHT],
}

struct ClassedImage {
    image: MNISTImage,
    class: u8,
}

struct TrainingData {
    dataset: Vec<ClassedImage>,
}

impl MNISTImage {
    fn blank() -> MNISTImage {
        MNISTImage {
            data: [0; 28 * 28],
        }
    }

    fn from_file(filename: &str) -> Result<MNISTImage, String> {
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

impl TrainingData {
    fn new() -> TrainingData {
        TrainingData {
            dataset: Vec::new()
        }
    }

    fn from_directory(directory: &str) -> Result<TrainingData, String> {
        let training_directory = directory.to_owned() + "/train/";
        
        let mut new_trainingdata = TrainingData::new();

        for digit in 0..=9 {
            //Create a new string (cloned from the general training string) to point to this digit's subfolder
            let current_directory = training_directory.clone() + &digit.to_string();

            //Get a reader to list all entries inside current_directory
            let reader = match fs::read_dir(&current_directory) {
                Ok(unpacked_reader) => unpacked_reader,
                Err(_) => return Err(format!("Cannot open {} for reading", current_directory)),
            };

            for entry in reader {
                //Check whether entry exists
                let entry = match entry {
                    Ok(unpacked_entry) => unpacked_entry,
                    Err(_) => return Err(format!("Error unpacking entry under {}", current_directory)),
                };

                //Convert the DirEntry to a string path to the file in question
                let path = match entry.path().to_str() {
                    Some(unpacked_path) => unpacked_path.to_string(), //.to_str gives a slice which will be lost when the result of entry.path gets deallocated, so we convert it to a String
                    None => return Err(format!("Cannot convert path {} to UTF-8 String", entry.path().display())),
                };

                //Read in the image data
                let img = match MNISTImage::from_file(&path) {
                    Ok(unpacked_image) => unpacked_image,
                    Err(string) => return Err(string), //Our functions are already set up to return user-readable Strings, so we don't need to make one up like we did for the external functions
                };

                //Add the new image to the end of the new dataset
                new_trainingdata.dataset.push(ClassedImage {
                    image: img,
                    class: digit,
                });
            }
        }

        Ok(new_trainingdata)
    }
}

const DATASET_DIRECTORY: &str = "minimalmnist";

fn main() {
    let dataset = match TrainingData::from_directory(DATASET_DIRECTORY) {
        Ok(dataset) => dataset,
        Err(msg) => {
            println!("Error loading data: {}",msg);
            return;
        }
    };
    println!("Data imported successfully! Length: {}", dataset.dataset.len());
}
