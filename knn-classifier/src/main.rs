use image::{ImageReader, DynamicImage, GenericImageView, Pixel, Rgb};
use std::io::BufReader;
use std::fs;
use rayon::prelude::*;
use std::time::Instant;
use std::cmp::Ordering;

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

    fn parallel_calculate_distance(&self, other: &MNISTImage) -> f64 {
        let accumulator: f64 = self.data.par_iter().zip(other.data.par_iter()).map(|(a, b)| ((*a as i16 - *b as i16) as f64).powf(2f64)).sum();
        return accumulator.sqrt()
    }

    fn calculate_distance(&self, other: &MNISTImage) -> f64 {
        let accumulator: f64 = self.data.iter().zip(other.data.iter()).map(|(a, b)| ((*a as i16 - *b as i16) as f64).powf(2f64)).sum();
        return accumulator.sqrt()
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

            let digitset: Vec<Result<MNISTImage, String>> = reader.map(|entry| Self::entry_to_img(&entry, &current_directory)).collect();
            for img_result in digitset {
                match img_result {
                    Ok(img) => new_trainingdata.dataset.push(ClassedImage {
                        image: img,
                        class: digit,
                    }),
                    Err(msg) => return Err(msg),
                };
            }
        }

        Ok(new_trainingdata)
    }

    fn parallel_from_directory(directory: &str) -> Result<TrainingData, String> {
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

            let reader: Vec<Result<fs::DirEntry, std::io::Error>> = reader.collect();
            let reader = reader.par_iter();
            let digitset: Vec<Result<MNISTImage, String>> = reader.map(|entry| Self::entry_to_img(entry, &current_directory)).collect();
            for img_result in digitset {
                match img_result {
                    Ok(img) => new_trainingdata.dataset.push(ClassedImage {
                        image: img,
                        class: digit,
                    }),
                    Err(msg) => return Err(msg),
                };
            }
        }

        Ok(new_trainingdata)
    }

    fn entry_to_img(entry: &Result<fs::DirEntry, std::io::Error>, current_directory: &str) -> Result<MNISTImage, String> {
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
        match MNISTImage::from_file(&path) {
            Ok(unpacked_image) => Ok(unpacked_image),
            Err(string) => return Err(string), //Our functions are already set up to return user-readable Strings, so we don't need to make one up like we did for the external functions
        }
    }

    fn classify(&self, image: &MNISTImage, k: u32) -> Result<u8, String> {
        let k = if let Ok(k) = usize::try_from(k) { k } else { return Err(format!("k of {} is larger than maximum array size on this platform.", k)) };
        let mut lowest_distances = vec![(f64::INFINITY,0u8); k];

        for training_img in &self.dataset {
            let distance = image.calculate_distance(&training_img.image);
            place_in_vector(distance, &mut lowest_distances, training_img.class);
        }

        Ok(take_votes(lowest_distances))
    }

    fn parallel_classify(&self, image: &MNISTImage, k: u32) -> Result<u8, String> {
        let k = if let Ok(k) = usize::try_from(k) { k } else { return Err(format!("k of {} is larger than maximum array size on this platform.", k)) };

        let mut distances: Vec<(f64, u8)> = self.dataset.par_iter().map(|x| (image.calculate_distance(&x.image), x.class)).collect();

        distances.par_sort_by(|a, b| float_compare(a.0, b.0));
        distances.truncate(k);

        Ok(take_votes(distances))
    }
}

fn float_compare(a: f64, b: f64) -> Ordering{
    a.partial_cmp(&b).expect(&format!("{} and {} cannot be compared", a, b))
}

fn take_votes(lowest_distances: Vec<(f64, u8)>) -> u8{
    const NUM_CLASSES: u8 = 10;
    let mut votes = [0; NUM_CLASSES as usize];
    for (dist, class) in lowest_distances {
        votes[usize::from(class)] += 1;
    }

    let mut highest = (0, 0);
    for i in 0..NUM_CLASSES {
        let this_vote = votes[usize::from(i)];
        if this_vote > highest.0 {
            highest = (this_vote, i);
        }
    }
    highest.1
}

fn place_in_vector(val: f64, vector: &mut Vec<(f64, u8)>, class: u8) {
    if val >= vector[0].0 {
        return;
    }
    let len = vector.len();
    for i in 1..len {
        vector[i - 1] = vector[i];
        if val > vector[i].0 {
            vector[i - 1] = (val, class);
            return;
        }
    }
    vector[len - 1] = (val, class);
}

fn test(model: TrainingData, k: u32, data_directory: &str, verbose: bool) -> Result<f64, String> {
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

            let t0 = Instant::now();
            let class = match model.parallel_classify(&img, k) {
                Ok(class) => class,
                Err(string) => return Err(string),
            };
            println!("classed in {}s", t0.elapsed().as_secs_f64());

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

const DATASET_DIRECTORY: &str = "mnist_png";

fn main() {
    let thread_count = 4;
    let threads_beyond_main = thread_count - 1
    rayon::ThreadPoolBuilder::new().num_threads(threads_beyond_main).build_global(); //set maximum threads across all rayon pools

    let t0 = Instant::now();
    let dataset = match TrainingData::parallel_from_directory(DATASET_DIRECTORY) {
        Ok(dataset) => dataset,
        Err(msg) => {
            println!("Error loading data: {}",msg);
            return;
        }
    };
    println!("loaded in {}s", t0.elapsed().as_secs_f64());
    println!("Data imported successfully! Length: {}", dataset.dataset.len());

    println!("Beginning testing ...");
    match test(dataset, 4, DATASET_DIRECTORY, true) {
        Ok(score) => println!("Test complete.\nAccuracy: {}%", score * 100.0),
        Err(string) => println!("Error while testing: {}",string),
    }
}
