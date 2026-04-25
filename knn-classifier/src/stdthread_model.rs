use std::fs;
use std::thread;
//use std::vec;
use std::mem::{swap};
use std::sync::mpsc::{channel, Receiver, Sender};
use crate::mnist_image::{MNISTImage, ClassedImage};
use crate::knn_model::{KNNModel};

// This is the best implimentation I can think of without mangling the rest of the project code
static NUMBER_OF_THREADS: usize = 10;

pub struct StdThreadModel {
    dataset: Vec<ClassedImage>,
    number_of_threads: usize,
    threads: Vec<(Option<thread::JoinHandle<()>>, Sender<Option<MNISTImage>>)>,
    return_channel: Receiver<(f64, u8)>,
    return_channel_sender: Sender<(f64, u8)>,
}

impl StdThreadModel {

    fn initialize(&mut self) {
        
        for thread_id in 0..self.number_of_threads {

            //println!("\n Spawning thread {} \n responsable for training data slice {}-{}", thread_id, (thread_id*(self.len()/self.number_of_threads)), (thread_id+1)*(self.len()/self.number_of_threads));

            // get slice of data set for thread at thread_id
            let dataset_slice: Vec<ClassedImage> = if thread_id == self.number_of_threads - 1 {
                self.dataset[thread_id*(self.len()/self.number_of_threads)..self.len()].to_vec()
            } else {
                self.dataset[thread_id*(self.len()/self.number_of_threads)..(thread_id+1)*(self.len()/self.number_of_threads)].to_vec()
            };
            
            let (working_image_tx, working_image_rx) = channel::<Option<MNISTImage>>();
            let local_return_channel_sender = self.return_channel_sender.clone();
            let local_thread_id = thread_id;


            let handle = thread::spawn( move || {
                println!("Spawned {}, covering slice size: {}", thread_id, dataset_slice.len());
                loop{
                    //println!("thread {}, awaiting", thread_id);
                    let mut test_image: MNISTImage;

                    match working_image_rx.recv() {
                        Ok(message) => {
                            match message {
                                Some(image) => test_image = image,
                                None => {
                                    break
                                },
                            }
                        },
                        Err(_) => break,
                    }
                    //println!("thread {}, got image", thread_id);

                    for training_image in &dataset_slice {
                        local_return_channel_sender.send((test_image.calculate_distance(&training_image.image), training_image.class));
                    }

                    //println!("thread {}, done with image", thread_id);
                    local_return_channel_sender.send((f64::INFINITY, 255 )); // send impossible result as marker for end of thread process 
                }
                println!("Thread {} dropping", thread_id);
            });

            self.threads.push((Some(handle), working_image_tx));
        }
    }

}

impl Drop for StdThreadModel{
    fn drop (&mut self){
        for mut thread in &mut self.threads{
            thread.1.send(None);

            let mut handle: Option<thread::JoinHandle<()>> = None;

            swap(&mut handle, &mut thread.0);

            match handle {
                Some(join_handle) => {
                    if !join_handle.is_finished() {
                        
                    }
                    join_handle.join()
                },
                None => continue,
            };
        }
    }
}

impl KNNModel for StdThreadModel {
    fn new() -> StdThreadModel {
        let (return_channel_tx, return_channel_rx) = channel::<(f64, u8)>();
        return StdThreadModel {
            dataset: Vec::new(),
            number_of_threads: NUMBER_OF_THREADS,
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
        
        new_model.initialize();

        Ok(new_model)
    }

    fn classify(&self, image: &MNISTImage, k: u32) -> Result<u8, String> {
        
        
        //println!("Sending image");

        for (_, input_channel) in &self.threads {
            input_channel.send(Some(image.clone()));
        }


        let mut distances: Vec<(f64, u8)> = vec![(f64::INFINITY, 255); usize::try_from(k).unwrap()];
        let mut threads_complete: usize = 0;
        
        loop {

            // close loop after all threads are done with work
            if threads_complete >= self.number_of_threads{
                break;
            }

            let message = self.return_channel.recv();

            match message {
                Ok((distance, class)) => {
                    if distance == f64::INFINITY {
                        threads_complete += 1;
                        //println!("Threads complete: {} of {}", threads_complete, self.number_of_threads);
                        continue; 
                    }
                    for i in 0..k {
                        if distance < distances[usize::try_from(i).unwrap()].0 {
                            distances.insert(usize::try_from(i).unwrap(), (distance, class));
                        }
                    }
                },
                Err(err) => println!("Failed to read response from thread, error: {}", err),
            }

            //println!("\n\n{:?}", distances);


        }
        

        distances.truncate(usize::try_from(k).unwrap());
        //println!("\n\n{:?}", distances);


        return Ok(Self::take_votes(distances))
    }
}