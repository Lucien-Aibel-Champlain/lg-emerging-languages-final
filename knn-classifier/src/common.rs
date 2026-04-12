use image::{ImageReader, DynamicImage, GenericImageView, Pixel, Rgb};
use std::io::BufReader;
use std::fs;
use std::collections::HashMap;

pub const MNIST_WIDTH: usize = 28;
pub const MNIST_HEIGHT: usize = MNIST_WIDTH;

pub struct MNISTImage {
    pub data: [u8; MNIST_WIDTH * MNIST_HEIGHT],
}

pub struct ClassedImage {
    pub image: MNISTImage,
    pub class: u8,
}

pub struct TrainingData {
    pub dataset: Vec<ClassedImage>,
}

struct DistanceFinder {
    lowest_distances: Vec<(f64, u8)>,
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

    pub fn from_imagereader(reader: ImageReader<BufReader<fs::File>>) -> Result<MNISTImage, String> {
        match reader.decode() {
            Ok(imagedata) => Self::from_dynamicimage(imagedata),
            Err(_) => Err("Error decoding image file.".to_string()),
        }
    }

    pub fn average_pixel_channels(pix: Rgb<u8>) -> u8 {
        let mut sum: u16 = 0;
        for i in 0..=2 {
            sum += u16::from(pix[i]);
        }

        match u8::try_from(sum / 3) {
            Ok(avg) => avg,
            Err(_) => panic!("Tried to average pixel ({},{},{}), found value greater than 255.", pix[0],pix[1],pix[2]),
        }
    }

    pub fn from_dynamicimage(image_data: DynamicImage) -> Result<MNISTImage, String> {
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
            output += &((value as u8).to_string() + " ");
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
    pub fn new() -> TrainingData {
        TrainingData {
            dataset: Vec::new()
        }
    }

    pub fn from_directory(directory: &str) -> Result<TrainingData, String> {
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

    fn classify(&self, image: &MNISTImage, k: u32) -> Result<u8, String> {
        let k = if let Ok(k) = usize::try_from(k) { k } else { return Err(format!("k of {} is larger than maximum array size on this platform.", k)) };
        let mut distance_list = DistanceFinder::new(k);

        for training_img in &self.dataset {
            distance_list.ordered_insert(image.calculate_distance(&training_img.image), training_img.class);
        }

        match distance_list.count_votes() {
            Some(class) => Ok(class),
            None => Err("No training data found to compare to.".to_string()),
        }
    }
}

impl DistanceFinder {
    fn new(max_size: usize) -> DistanceFinder {
        DistanceFinder {
            lowest_distances: vec![(f64::INFINITY,0u8); max_size],
        }
    }

    fn ordered_insert(&mut self, val: f64, class: u8) {
        if val >= self.lowest_distances[0].0 {
            return;
        }
        let len = self.lowest_distances.len();
        for i in 1..len {
            self.lowest_distances[i - 1] = self.lowest_distances[i];
            if val > self.lowest_distances[i].0 {
                self.lowest_distances[i - 1] = (val, class);
                return;
            }
        }
        self.lowest_distances[len - 1] = (val, class);
    }

    fn count_votes(&self) -> Option<u8> {
        let mut votes: HashMap<u8, u32> = HashMap::new();
        for (_dist, class) in &self.lowest_distances {
            votes.entry(*class).and_modify(|num| { *num += 1 }).or_insert(1);
        }

        self.find_highest(votes.iter())
    }

    fn find_highest<'a>(&self, mut iterator: impl Iterator<Item=(&'a u8, &'a u32)>) -> Option<u8> {
        let mut lowest = if let Some(pair) = iterator.next() { (*pair.0, *pair.1) } else { return None };
        for (class, votes) in iterator {
            if *votes > lowest.1 {
                lowest = (*class, *votes);
            }
        }
        Some(lowest.0)
    }
}

pub fn test(model: TrainingData, k: u32, data_directory: &str, verbose: bool) -> Result<f64, String> {
    let mut successes = 0;
    let mut total = 0;
    let test_directory = data_directory.to_owned() + "/test/";

    for digit in 0..=9 {
        //Create a new string (cloned from the general training string) to point to this digit's subfolder
        let current_directory = test_directory.clone() + &digit.to_string();

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

            let class = match model.classify(&img, k) {
                Ok(class) => class,
                Err(string) => return Err(string),
            };

            if class == digit {
                successes += 1;
            }
            if verbose {
                match class == digit {
                    true => println!("Succesfully classed {}.", path),
                    false => println!("MISS on test case {}. Expected result {}, got {}.", path, digit, class),
                }
            }
            total += 1;
        }
    }

    Ok(f64::from(successes) / f64::from(total))
}