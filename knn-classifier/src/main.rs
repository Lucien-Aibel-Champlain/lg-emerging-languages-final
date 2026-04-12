use crate::sequential::{SequentialKNN};

pub mod sequential;

const DATASET_DIRECTORY: &str = "minimalmnist";

fn main() {
    let dataset: SequentialKNN = match SequentialKNN::from_directory(DATASET_DIRECTORY) {
        Ok(dataset) => dataset,
        Err(msg) => {
            println!("Error loading data: {}",msg);
            return;
        }
    };
    println!("Data imported successfully! Length: {}", dataset.len());

    println!("Beginning testing ...");
    match dataset.test(4, DATASET_DIRECTORY, true) {
        Ok(score) => println!("Test complete.\nAccuracy: {}%", score * 100.0),
        Err(string) => println!("Error while testing: {}",string),
    }
}
