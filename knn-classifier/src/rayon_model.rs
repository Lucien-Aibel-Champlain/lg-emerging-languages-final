use std::fs;
use rayon::prelude::*;
use crate::mnist_image::{MNISTImage, ClassedImage};
use crate::knn_model::{KNNModel};

pub struct RayonModel {
    dataset: Vec<ClassedImage>,
}

impl KNNModel for RayonModel {
    #[allow(refining_impl_trait)]
    fn new() -> RayonModel {
        RayonModel {
            dataset: Vec::new()
        }
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }

    #[allow(refining_impl_trait)]
    fn from_directory(directory: &str) -> Result<RayonModel, String> {
        let training_directory = directory.to_owned() + "/train/";
        
        let mut new_model = RayonModel::new();

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
                    Ok(img) => new_model.dataset.push(ClassedImage {
                        image: img,
                        class: digit,
                    }),
                    Err(msg) => return Err(msg),
                };
            }
        }

        Ok(new_model)
    }
    
    fn classify(&self, image: &MNISTImage, k: u32) -> Result<u8, String> {
        let k = if let Ok(k) = usize::try_from(k) { k } else { return Err(format!("k of {} is larger than maximum array size on this platform.", k)) };

        let mut distances: Vec<(f64, u8)> = self.dataset.par_iter().map(|x| (image.calculate_distance(&x.image), x.class)).collect();

        distances.par_sort_by(|a, b| Self::float_compare(a.0, b.0));
        distances.truncate(k);

        Ok(Self::take_votes(distances))
    }
}