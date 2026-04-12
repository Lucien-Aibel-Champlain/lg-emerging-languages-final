use super::common::{MNIST_HEIGHT, MNIST_WIDTH, MNISTImage, TrainingData};

pub struct SequentialKNN {
    model: TrainingData,
}

impl SequentialKNN {
    pub fn from_directory(directory: &str) -> Result<SequentialKNN, String> {
        match TrainingData::from_directory(directory) {
            Ok(data) => Ok(SequentialKNN {
                model: data,
            }),
            Err(msg) => Err(msg),
        }
    }
    pub fn len(&self) -> usize {
        self.model.dataset.len()
    }

    pub fn test(&self, k: u32, data_directory: &str, verbose: bool) -> Result<f64, String> {
        self.model.test(k, data_directory, verbose)
    }
}

impl MNISTImage {
    fn calculate_distance(&self, other: &MNISTImage) -> f64 {
        let mut accumulator: f64 = 0f64;
        for i in 0..(MNIST_WIDTH * MNIST_HEIGHT){
            accumulator += ((self.data[i] as i16 - other.data[i] as i16) as f64).powf(2f64);
        }
        accumulator.sqrt()
    }
}