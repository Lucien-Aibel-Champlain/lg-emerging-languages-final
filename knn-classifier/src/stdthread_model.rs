use std::fs;
use std::thread;
//use std::vec;
use std::sync::mpsc::{channel, Receiver, Sender};
use crate::mnist_image::{MNISTImage, ClassedImage};
use crate::knn_model::{KNNModel};


pub struct StdThreadModel {
    dataset: Vec<ClassedImage>,
    threads: Vec<(thread::JoinHandle<()>, Sender<MNISTImage>)>,
    return_channel: Receiver<(f64, u8)>,
    return_channel_sender: Sender<(f64, u8)>,
}

impl StdThreadModel {

    fn initialize(&mut self, number_of_threads: usize) {
        

        for thread_id in 0..number_of_threads {
            let dataset_slice: Vec<ClassedImage> = self.dataset[thread_id*(self.len()/number_of_threads)..(thread_id+1)*(self.len()/number_of_threads)].to_vec();
            let (working_image_tx, working_image_rx) = channel::<MNISTImage>();
            let local_return_channel_sender = self.return_channel_sender.clone();
            


            let handle = thread::spawn( move || {
                loop{
                    let message = match working_image_rx.recv() {
                        Ok(test_image) => {
                            for training_image in &dataset_slice {
                                local_return_channel_sender.send((test_image.calculate_distance(&training_image.image), training_image.class));
                            }
                            continue;
                        },
                        Err(_) => panic!("Something broke!"),
                    };
                    local_return_channel_sender.send((f64::INFINITY, 255 ));
                }
            });

            self.threads.push((handle, working_image_tx));
        }
    }
}

impl KNNModel for StdThreadModel {
    fn new() -> StdThreadModel {
        let (return_channel_tx, return_channel_rx) = channel::<(f64, u8)>();
        return StdThreadModel {
            dataset: Vec::new(),
            threads: Vec::new(),
            return_channel: return_channel_rx,
            return_channel_sender: return_channel_tx,
        };
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }

    fn from_directory(directory: &str) -> Result<StdThreadModel, String> {
        let training_directory = directory.to_owned() + "/train/";
        
        let mut new_model = StdThreadModel::new();

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
                    Ok(img) => new_model.dataset.push(ClassedImage {
                        image: img,
                        class: digit,
                    }),
                    Err(msg) => return Err(msg),
                };
            }
        }
        
        new_model.initialize(10);

        Ok(new_model)
    }

    fn classify(&self, image: &MNISTImage, k: u32) -> Result<u8, String> {
        
        for (_, input_channel) in &self.threads {
            input_channel.send(image.clone());
        }

        let mut distances: Vec<(f64, u8)> = vec![(f64::INFINITY, 255); usize::try_from(k).unwrap()];
        loop{
            let message = match self.return_channel.recv() {
                Ok((distance, class)) => {
                    if distance == f64::INFINITY {
                        break;
                    }
                    for i in 0..k {
                        if distance < distances[usize::try_from(i).unwrap()].0 {
                            distances.insert(usize::try_from(i).unwrap(), (distance, class));
                        }
                    }
                },
                Err(_) => panic!(),
            };
        }
        return Ok(Self::take_votes(distances))
    }
}