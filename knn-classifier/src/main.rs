use crate::common::{MNIST_HEIGHT, MNIST_WIDTH, MNISTImage, TrainingData};

pub mod common;

impl MNISTImage {
    fn calculate_distance(&self, other: &MNISTImage) -> f64 {
        let mut accumulator: f64 = 0f64;
        for i in 0..(MNIST_WIDTH * MNIST_HEIGHT){
            accumulator += ((self.data[i] as i16 - other.data[i] as i16) as f64).powf(2f64);
        }
        accumulator.sqrt()
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

    println!("Beginning testing ...");
    match dataset.test(4, DATASET_DIRECTORY, true) {
        Ok(score) => println!("Test complete.\nAccuracy: {}%", score * 100.0),
        Err(string) => println!("Error while testing: {}",string),
    }
}
