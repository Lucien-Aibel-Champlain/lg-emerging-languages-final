use std::fs;
use std::time::Instant;
use crate::mnist_image::{MNISTImage};
use std::cmp::Ordering;

pub trait KNNModel {
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

    fn test(&self, k: u32, data_directory: &str, verbose: bool) -> Result<f64, String> {
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
                let class = match self.classify(&img, k) {
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

    fn float_compare(a: f64, b: f64) -> Ordering{
        a.partial_cmp(&b).expect(&format!("{} and {} cannot be compared", a, b))
    }

    fn take_votes(lowest_distances: Vec<(f64, u8)>) -> u8{
        const NUM_CLASSES: u8 = 10;
        let mut votes = [0; NUM_CLASSES as usize];
        for (_dist, class) in lowest_distances {
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

    fn load_and_test(directory: &str) {
        let t0 = Instant::now();
        let model = match Self::from_directory(directory) {
            Ok(model) => model,
            Err(msg) => {
                println!("Error loading data: {}",msg);
                return;
            }
        };
        let t0 = t0.elapsed().as_secs_f64();
        println!("Data imported successfully! Length: {}", model.len());
        println!("Loaded in {}s", t0);

        println!("Beginning testing ...");
        match model.test(6, directory, true) {
            Ok(score) => println!("Test complete.\nAccuracy: {}%", score * 100.0),
            Err(string) => println!("Error while testing: {}",string),
        };
    }

    fn new() -> impl KNNModel;
    fn len(&self) -> usize;
    fn from_directory(directory: &str) -> Result<impl KNNModel, String>;
    fn classify(&self, image: &MNISTImage, k: u32) -> Result<u8, String>;
}