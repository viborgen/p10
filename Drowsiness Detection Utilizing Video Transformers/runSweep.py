#imports
import os
from tqdm import tqdm
import random
import shutil
from huggingface_hub import notebook_login
import wandb
from collections import defaultdict
import pytorchvideo.data
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    UniformTemporalSubsample,
)
from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    Resize,
)
import numpy as np
from IPython.display import Image
from transformers import TrainingArguments, Trainer, EvalPrediction
import evaluate
import torch
from sklearn.metrics import confusion_matrix,precision_recall_curve
from transformers import EarlyStoppingCallback
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight
import torch.nn as nn
import csv
import sys
import random
import shutil

#General
video_length = 60
frame_rate = 10
projectName = "drowsinessSWEEP"
mapName = "utaFINALtest"

# train
num_epochs = 50
testingWandb = False #True if everything should be logged as "test"
resume_from_checkpoint = False #if true huggingface finds the latest checkpoint automatically
checkpoint_save_limit = 2 #None:saving all checkpoints, 2:saving best and last.
useClassWeights = True

#paths
groundFolderPath = "/work/drowsiness_experiment/"
origDataPath = groundFolderPath + "uta-reallife-drowsiness-dataset/" #path to the original data
producedVideoPath = groundFolderPath + "videos/facesFinal_sorted/" #path to store videos of the original data
reorganizedPath = groundFolderPath + "videos/videosSorted/" #path to store reorganized videos


#sorting datasaet using system links in Ubuntu and defined folds to follow while sorting
folds = [
    ['01','02','03','04','05','06','07','08','09','10','11','12'],
    ['13','14','15','16','17','18','19','20','21','22','23','24'],
    ['25','26','27','28','29','30','31','32','33','34','35','36'],
    ['37','38','39','40','41','42','43','44','45','46','47','48'],
    ['49','50','51','52','53','54','55','56','57','58','59','60']
]
def create_loso_symlinks(main_folder, output_folder, folds, video_length,skipSelector,frame_rate):
    """
    Create Leave-One-Subject-Out (LOSO) symlinks for training, validation, and test datasets.

    This function organizes datasets into training, validation, and test folders based on the 
    Leave-One-Subject-Out (LOSO) cross-validation technique. Symlinks are created instead of 
    copying files to save space and time.

    Parameters:
    main_folder (str): The path to the main directory containing the data.
    output_folder (str): The path to the directory where the symlinked data will be stored.
    folds (list of lists): A list where each sublist contains the subjects designated for a particular test fold.
    video_length (int): The length of the videos in the dataset.
    skipSelector (int): An integer value indicating how many frames to skip when creating training symlinks.
    frame_rate (int): The frame rate of the videos.

    Returns:
    None
    """
    # Check if the folder exists
    if os.path.exists(output_folder):
        # Delete the folder
        shutil.rmtree(output_folder)
    classes = ['0', '10']
    main_folder = main_folder + f'fps{frame_rate}/' +f"len{video_length}/"
    path = f'{main_folder}' 
    for i, test_fold in enumerate(folds):
        for class_name in classes:
            fold_folder = os.path.join(output_folder, f'Fold_{i+1}')
            test_folder = os.path.join(fold_folder, f'test', class_name)
            train_folder = os.path.join(fold_folder, f'train', class_name)
            validation_folder = os.path.join(fold_folder, f'val', class_name)
            os.makedirs(test_folder, exist_ok=True)
            os.makedirs(train_folder, exist_ok=True)
            os.makedirs(validation_folder, exist_ok=True)
            for subject_folder in os.listdir(main_folder):
                subject = os.path.basename(subject_folder)
                if subject in test_fold:
                    try:
                        # This is the test fold, so create symlinks in the test folder
                        for filename in os.listdir(os.path.join(path, subject_folder, class_name)):
                            os.symlink(os.path.join(path, subject_folder, class_name, filename), os.path.join(test_folder, filename))
                    except:
                        print(f"Error processing {subject_folder}/{class_name}")
                else:
                    # This is a training fold, so create symlinks in the train and validation folders
                    try:
                        files = os.listdir(os.path.join(path, subject_folder, class_name))
                        random.shuffle(files)  # Shuffle the files
                        train_files = files[:len(files)*9//10]  # Use 90% of the files for training
                        validation_files = files[len(files)*9//10:]  # Use the remaining 10% of the files for validation

                        for idx, filename in enumerate(train_files):
                            if idx % skipSelector == 0:
                                os.symlink(os.path.join(path, subject_folder, class_name, filename), os.path.join(train_folder, filename))

                        for filename in validation_files:
                            os.symlink(os.path.join(path, subject_folder, class_name, filename), os.path.join(validation_folder, filename))
                    except:
                        print(f"Error processing {subject_folder}/{class_name}")

#loading the data for use in the model as well as defining the model
def loadData(root_folder):
    """
    Load video data and initialize a pre-trained video classification model.

    This function scans the root folder for video files, determines class labels, and initializes
    datasets for training, validation, and testing. It also configures a pre-trained 
    Timesformer model for video classification and optionally applies class weights for 
    imbalanced classes.
    
    Parameters:
    root_folder (str): The path to the root directory containing the video data organized 
                       in subfolders by class and split into train, val, and test sets.

    Returns:
        - model_ckpt (str): The checkpoint identifier for the pre-trained model.
        - train_dataset (pytorchvideo.data.Ucf101): The training dataset.
        - val_dataset (pytorchvideo.data.Ucf101): The validation dataset.
        - model (TimesformerForVideoClassification): The initialized video classification model.
        - image_processor (AutoImageProcessor): The image processor for preprocessing video frames.
        - test_dataset (pytorchvideo.data.Ucf101): The test dataset.
        - label2id (dict): Mapping from class labels to numerical IDs.
        - id2label (dict): Mapping from numerical IDs to class labels.
        - amountOfData (list): Number of videos in train, validation, and test datasets.
        - class_count (dict): Count of videos in each class.
        - class_weights (numpy.ndarray or None): Computed class weights for the training set if 
                                                 `useClassWeights` is True, otherwise None.

    """
    def get_all_video_paths(root_folder, video_extensions=['.mp4', '.avi', '.mov']):
        """
        Get all video file paths in a folder, including subfolders.
        Args:
            root_folder (str): Root folder to start the search.
            video_extensions (list): List of video file extensions to search for.

        Returns:
            List[str]: List of video file paths.
        """
        video_paths = []
        for foldername, subfolders, filenames in os.walk(root_folder):
            for filename in filenames:
                if any(filename.endswith(ext) for ext in video_extensions):
                    video_paths.append(os.path.join(foldername, filename))
        return video_paths
    all_video_file_paths = get_all_video_paths(root_folder)

    class_labels = sorted({str(path).split("/")[-2] for path in all_video_file_paths})
    label2id = {label: i for i, label in enumerate(class_labels)}
    print('label2id',label2id)
    global id2label
    id2label = {i: label for label, i in label2id.items()}
    print('id2label', id2label)

    print(f"Unique classes: {list(label2id.keys())}.")
        # Initialize a dictionary to count elements in each class
    from collections import defaultdict
    class_count = defaultdict(int)

    print('Total class count:')
    for path in all_video_file_paths:
        class_label = str(path).split("/")[-2]
        class_count[class_label] += 1
    for class_label, count in sorted(class_count.items()):
        print(f"Class '{class_label}': {count} video{'s' if count != 1 else ''}")

    train_class_count = defaultdict(int)
    val_class_count = defaultdict(int)
    test_class_count = defaultdict(int)
    #calculate classweights for train
    for path in all_video_file_paths:
        if "train/" in str(path):
            class_label = str(path).split("/")[-2]
            train_class_count[class_label] += 1
        if "val/" in str(path):
            class_label = str(path).split("/")[-2]
            val_class_count[class_label] += 1
        if "test/" in str(path):
            class_label = str(path).split("/")[-2]
            test_class_count[class_label] += 1
    
    # Print the counts for each class
    print('Splitted class count:')
    global class_dist
    class_dist = {}
    for class_label in sorted(train_class_count.keys()):
        print(f"Class '{class_label}': train: {train_class_count[class_label]}, val: {val_class_count[class_label]}, test: {test_class_count[class_label]}")
        class_dist[f"Stats/Class '{class_label}'"] = {'Train': train_class_count[class_label], 'Validation': val_class_count[class_label], 'Test': test_class_count[class_label]}
    if(useClassWeights == True):
        y = [label for label, count in train_class_count.items() for _ in range(count)]
        from sklearn.utils.class_weight import compute_class_weight
        import numpy as np
        class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
        print(f'Class weights: {class_weights}')
    else:
        class_weights = None
    
    #model selection
    from transformers import AutoImageProcessor, TimesformerForVideoClassification
    model_ckpt = "facebook/timesformer-base-finetuned-k400"
    image_processor = AutoImageProcessor.from_pretrained("facebook/timesformer-base-finetuned-k400")
    model = TimesformerForVideoClassification.from_pretrained(
        model_ckpt,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True, 
        num_frames=num_frames,
        hidden_dropout_prob = drop_out,
        attention_probs_dropout_prob = drop_out
    )

    if freeze:
        # Freeze all the parameters in the model
        for param in model.parameters():
            param.requires_grad = False
        
        num_layers = len(model.timesformer.encoder.layer)  # Get the number of layers
        num_layers_to_unfreeze = 1  # Or whatever number you want
        layers_to_unfreeze = [num_layers - i for i in range(1, num_layers_to_unfreeze + 1)]
        #layers_to_unfreeze = [num_layers - 2, num_layers - 1]  # Last two layers

        for i in layers_to_unfreeze:
            for param in model.timesformer.encoder.layer[i].parameters():
                param.requires_grad = True

        # You may also want to make sure the classifier layer remains trainable
        for param in model.classifier.parameters():
            param.requires_grad = True


    #defining dataset characteristics
    mean = image_processor.image_mean
    std = image_processor.image_std
    if "shortest_edge" in image_processor.size:
        height = width = image_processor.size["shortest_edge"]
    else:
        height = image_processor.size["height"]
        width = image_processor.size["width"]
    resize_to = (height, width)

    #model.config.num_frames = num_frames
    num_frames_to_sample = model.config.num_frames
    print(f'Looking at {num_frames_to_sample} frames in {video_length} s clips')
    clip_duration = video_length

    #augmenting and loading data
    train_transform = Compose(
        [
            ApplyTransformToKey(
                key="video",
                transform=Compose(
                    [
                        UniformTemporalSubsample(num_frames_to_sample),
                        Lambda(lambda x: x / 255.0),
                        Normalize(mean, std),
                        Resize((240,240)), #min size of the original video
                        RandomCrop(resize_to), #random crops to 224x224 to make sure camera movement is not a problem
                    ]
                ),
            ),
        ]
    )

    train_dataset = pytorchvideo.data.Ucf101(
        data_path=os.path.join(root_folder, "train"),
        clip_sampler=pytorchvideo.data.make_clip_sampler("random", clip_duration),
        decode_audio=False,
        transform=train_transform,
    )

    val_transform = Compose(
        [
            ApplyTransformToKey(
                key="video",
                transform=Compose(
                    [
                        UniformTemporalSubsample(num_frames_to_sample),
                        Lambda(lambda x: x / 255.0),
                        Normalize(mean, std),
                        Resize(resize_to),
                    ]
                ),
            ),
        ]
    )

    val_dataset = pytorchvideo.data.Ucf101(
        data_path=os.path.join(root_folder, "test"),
        clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", clip_duration),
        decode_audio=False,
        transform=val_transform,
    )

    test_dataset = pytorchvideo.data.Ucf101(
        data_path=os.path.join(root_folder, "test"),
        clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", clip_duration),
        decode_audio=False,
        transform=val_transform,
    )
    print(f'Amount of train:{train_dataset.num_videos}, val:{val_dataset.num_videos}, test:{test_dataset.num_videos}')
    amountOfData = [train_dataset.num_videos,val_dataset.num_videos,test_dataset.num_videos]
    return model_ckpt, train_dataset, val_dataset, model, image_processor, test_dataset, label2id, id2label,amountOfData,class_count,class_weights


#Defining trainer
def trainerDefinitions(model_ckpt, train_dataset, val_dataset, model, image_processor,folder, class_weights):
    """
    Defines the configuration and parameters for training a TimeSformer model.

    This function sets up the necessary components and arguments for training, including
    early stopping, training arguments, metrics computation, data collation, and an optional
    custom trainer class for handling class weights.

    Parameters:
    model_ckpt (str): Checkpoint identifier for the pre-trained model.
    train_dataset (Dataset): The dataset used for training.
    val_dataset (Dataset): The dataset used for validation.
    model (TimesformerForVideoClassification): The pre-trained video classification model.
    image_processor (AutoImageProcessor): Processor for preprocessing video frames.
    folder (str): Identifier for the training session.
    class_weights (numpy.ndarray or None): Class weights for handling imbalanced classes.
    
    Returns:
    tuple: Containing the following elements:
        - trainer (Trainer or CustomTrainer): The configured trainer object.
        - run_name (str): Name of the training run for logging and tracking.
    """
    model_name = model_ckpt.split("/")[-1]
    global new_model_name
    new_model_name = f"{model_name}-finetuned-{folder}-b{batch_size}-f{num_frames}-l{video_length}"
    if testingWandb == True:
        run_name = f'TEST'
    else:
        run_name=f'{str(folder)}_b{str(batch_size)}_e{str(num_epochs)}_f{num_frames}_l{video_length}_{model_name[:2]}'

    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=3,  # Number of evaluations with no improvement after which training will be stopped
        early_stopping_threshold=0.0, #0.00 # Minimum change in the monitored metric to qualify as an improvement
    )

    args = TrainingArguments(
        "checkpoints/"+new_model_name,
        remove_unused_columns=False,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        #eval_steps = 100,#(train_dataset.num_videos // batch_size) // 2,
        save_total_limit = checkpoint_save_limit,
        learning_rate=5e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_ratio=0.1,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=False,
        max_steps=(train_dataset.num_videos // batch_size) * num_epochs,
        report_to="wandb",
        run_name=run_name,
        weight_decay=weight_decay,
        fp16=True,
    )
    print("train_dataset.num_videos", train_dataset.num_videos)
    print('WEIGHT DECAY',weight_decay)
    metric = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        predictions = np.argmax(eval_pred.predictions, axis=1)
        return metric.compute(predictions=predictions, references=eval_pred.label_ids)

    # collate function
    def collate_fn(examples): 
        # permute to (num_frames, num_channels, height, width)
        pixel_values = torch.stack(
            [example["video"].permute(1, 0, 2, 3) for example in examples]
        )
        labels = torch.tensor([example["label"] for example in examples])
        return {"pixel_values": pixel_values, "labels": labels}
    
    class CostumTrainer(Trainer): #only used if class weights are used
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.get("logits")
            #loss_fct = nn.CrossEntropyLoss()#weight=torch.tensor(classWeights).to(model.device))
            loss_fct = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, device=model.device).float())
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

    
    trainer_params = {
        "model": model,
        "args": args,
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset,
        "tokenizer": image_processor,
        "compute_metrics": compute_metrics,
        "data_collator": collate_fn,
        "callbacks": [early_stopping_callback],
    }
    trainer_class = CostumTrainer if useClassWeights else Trainer
    print(f"Using trainer class: {trainer_class.__name__}")

    trainer = trainer_class(**trainer_params)
    return trainer, run_name

def train(trainer):
    """
    Making sure that the model can resume from a previous checkpoint, if it exists.
    """
    print(f'GPU is found: {torch.cuda.is_available()}')
    try:
        train_results = trainer.train(resume_from_checkpoint = resume_from_checkpoint)
    except FileNotFoundError:
        print("Checkpoint not found")
        train_results = trainer.train(resume_from_checkpoint = False)
    return train_results

def testing(test_dataset,trainer,model,label2id,runName):
    """
    Perform testing on a trained video classification model.

    This function runs inference on a test dataset using a trained model,
    computes various evaluation metrics, and logs the results to Weights & Biases.
    """
    print("Running testing")
    def run_inference(model, video):
        perumuted_sample_test_video = video.permute(1, 0, 2, 3)
        inputs = {
            "pixel_values": perumuted_sample_test_video.unsqueeze(0)
        }
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        model = model.to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        return logits
    test_size = test_dataset.num_videos
    trained_model = trainer.model
    true_labels = []
    predicted_labels = []
    model_true_labels = []
    model_predicted_labels = []
    available_true_class_label = []
    correct_predictions = 0
    model_correct_predictions = 0
    probabilities = 0
    predicted_probabilities = []
    # Create the directory if it does not exist
    if not os.path.exists('/home/ubuntu/work/drowsiness_experiment/runCSVs'):
        os.makedirs('/home/ubuntu/work/drowsiness_experiment/runCSVs')
    with open(f'/home/ubuntu/work/drowsiness_experiment/runCSVs/{runName}.csv', 'w', newline='') as csvfile:
        fieldnames = ['video_name', 'probabilities', 'true_label', 'predicted_label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in tqdm(range(test_size)):
            sample_test_video = next(iter(test_dataset))
            video_input = sample_test_video["video"]
            true_label = sample_test_video["label"]
            video_name = sample_test_video["video_name"]

            logits = run_inference(trained_model, video_input)
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            predicted_probabilities.append(probabilities.cpu().numpy())
            predicted_class_idx = logits.argmax(-1).item()
            true_labels.append(true_label)
            predicted_labels.append(predicted_class_idx)
            if predicted_class_idx == true_label:
                correct_predictions += 1
            #print(f"True label: {true_label}, Predicted label: {predicted_class_idx}")
            model_predicted_class_label = trained_model.config.id2label[predicted_class_idx]
            available_true_class_label = int(id2label[true_label])
            model_true_labels.append(available_true_class_label)
            model_predicted_labels.append(int(model_predicted_class_label))
            if int(model_predicted_class_label) == available_true_class_label:
                model_correct_predictions += 1
            #writer.writerow({'video_name': video_name, 'probabilities': probabilities.cpu().numpy(), 'true_label': available_true_class_label, 'predicted_label': int(model_predicted_class_label)})
    accuracy = (model_correct_predictions / test_size)
    print("Test/Test Accuracy: {:.4f}".format(accuracy))

    wandb.log({"Test/Test-accuracy": accuracy})

    conf_matrix = confusion_matrix(model_true_labels, model_predicted_labels)
    print(model_true_labels)
    print(model_predicted_labels)
    model_true_labels2 = [label2id[str(label)] for label in model_true_labels]
    model_predicted_labels2 = [label2id[str(label)] for label in model_predicted_labels]
    wandb.log({"Test/conf_mat" : wandb.plot.confusion_matrix(probs=None,y_true=model_true_labels2, preds=model_predicted_labels2, class_names=list(label2id.keys()))}) #
    predicted_probabilities_positive = [prob[0][1] for prob in predicted_probabilities]
    predicted_probabilities_2d = [[1 - prob[0][1], prob[0][1]] for prob in predicted_probabilities]

    wandb.log({'Un Test/roc': wandb.plots.ROC(model_true_labels2, predicted_probabilities_2d, list(label2id.keys()))})
    wandb.log({'Un Test/pr': wandb.plots.precision_recall(model_true_labels2, predicted_probabilities_2d, list(label2id.keys()))})
    sns.heatmap(conf_matrix, 
                annot=True,
                fmt='g', 
                xticklabels=label2id.keys(),
                yticklabels=label2id.keys())
    plt.ylabel('Prediction',fontsize=13)
    plt.xlabel('Actual',fontsize=13)
    plt.title('Confusion Matrix',fontsize=17)
    plt.savefig("confusion.png")
    #plt.show()

    wandb.log({"Un Test/Confusion Matrix": wandb.Image("confusion.png")})
    plt.clf()



def trainTest(config=None):
    """
    Train and test a video classification model using the specified configuration.

    This function initializes a Weights & Biases run and conducts training and testing
    for a video classification model. It organizes videos, loads data, defines the model,
    trains the model, and evaluates its performance while logging to Weights and Biases.
    """
    torch.cuda.empty_cache()
    with wandb.init(config=config, project="drowsinessSWEEP"): #disable if not sweep
        try:
            config = wandb.config
            global video_length
            video_length = config['video_length']
            global num_frames
            num_frames = config['frames']
            global batch_size
            batch_size = config['batch_size']
            global skipSelector
            skipSelector = config['skipSelector']
            global weight_decay 
            weight_decay = config['weight_decay']
            global drop_out
            drop_out = config['drop_out']
            groundFolderPath = "/home/ubuntu/work/drowsiness_experiment/"
            producedVideoPath = groundFolderPath + "videos/facesFinal_sorted/" #path to store videos of the original data
            reorganizedPath = groundFolderPath + f"videos/videosSorted/{video_length}/" #path to store reorganized videos
            create_loso_symlinks(producedVideoPath, reorganizedPath, folds, video_length, skipSelector, 10)
            reorganizedPath = groundFolderPath + f"videos/videosSorted/{video_length}/" #path to store reorganized videos
            rootFolders = [folder for folder in os.listdir(reorganizedPath) if os.path.isdir(os.path.join(reorganizedPath, folder))]
            print(f'To be trained and tested {rootFolders}')
            test_acc_list = []
            index = 0
            folder = rootFolders[index]
            from IPython.display import clear_output
            for folder in sorted(rootFolders)[:1]:
                clear_output(wait=True)
                print(f'Training and testing {folder}')
                root_folder = reorganizedPath+folder
                model_ckpt, train_dataset, val_dataset, model, image_processor, test_dataset, label2id, id2label, amountOfData, class_count, class_weights = loadData(root_folder)
                trainer, runName = trainerDefinitions(model_ckpt, train_dataset, val_dataset, model, image_processor, folder, class_weights)
                wandb.init(group=mapName,project=projectName,name=runName)
                wandb.config.batch_size = batch_size
                train_results = train(trainer)
                test_acc_list+=[testing(test_dataset, trainer, model, label2id,runName)]
                wandb.log(class_dist)
                wandb.log({'Stats/train_num_videos': amountOfData[0], 'Stats/val_num_videos': amountOfData[1], 'Stats/test_num_videos':amountOfData[2]}) #logging amout of data to wandb
                wandb.log({str('stats/class' + class_label): count for class_label, count in sorted(class_count.items())}) #logging class count to wandb
                torch.cuda.empty_cache()
            wandb.finish()
        except Exception as e:
            print(e)
            print('here!')
            wandb.run.finish(exit_code=1)
            torch.cuda.empty_cache()
            sys.exit(0)


#trainTest(config)
# Define your sweep configuration
sweep_config = {
    'name': 'drowsinessSWEEP',
    'method': 'bayes', 
    'metric': {
      'name': 'Un Test/Unoffical-test-accuracy',
      'goal': 'maximize'#'minimize'   
    },
    'parameters': {
        'frames': {
            'values': [5,10,20,30,40,50,60,70,80,90,100]
        },
        'video_length': {
            'values': [5,10,20,30,60] 
        },
        'batch_size': {
            'values': [1,2,4,8,16,32] 
        },
        'skipSelector': {
            'values': [1,2,3,4,5,6,7,8,9,10] 
        },
        'weight_decay': {
            'min': 0.00001, 
            'max': 4.0  
        },
        'drop_out': {
            'min': 0.0,
            'max': 0.8
        }
    }
}

#freeze layers and train only the last
freeze = True

# Run the sweep
resumeSweep = False
if resumeSweep:
    sweep_id = '' #put sweep ID from W&B
else:
    # Initialize the sweep
    sweep_id = wandb.sweep(sweep_config, project="drowsinessSWEEP") 


wandb.agent(sweep_id, function=trainTest, count=2000)