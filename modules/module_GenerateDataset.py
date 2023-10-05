# importing packages
import numpy as np
import zipfile
import collections
import re

class fslDataset:
    def __init__(self) -> None:
        pass

    def list_files_from_zip_path(path):
        """ 
        List the files in each class of the dataset given a PATH with the zip file
        
        Arguments
        ----------
        path: str like
            path of the zip file saved in the same folder
        
        Returns
        ----------
        file_names:
            list of all files read in the file.zip, including folder names
        """
        file_names = []
        with zipfile.ZipFile(path, 'r') as zip:
            for file_info in zip.infolist():
                file_names.append(file_info.filename)
        return file_names
    
    def get_files_per_class(videos, folders):
        """ 
        Compile videos according to class
        
        Arguments
        ---------
        videos : array_like 
            filename of videos
        folders : array_like
            filename of folders

        Returns
        --------
        video_and_class : dictionary
            dictionary with class (key) and videos (values)
        """
        video_and_class = collections.defaultdict(list)

        # Extracts the class number from the folder name
        for folder in folders:
            class_key = int(re.findall(r'\d+', folder)[0])

            for video in videos:
                if folder in video:
                    video_and_class[class_key].append(video)
        return video_and_class
    
    def subset_data(files, numclass, numvideos=20):
        """ 
        Selects a subset of the complete videos dictionary containing the first num_class classes.

        Args:
        file_dictionary - dictionary of the complete videos with class
        num_class - number of classes in the subset

        Return:
        files_subset - A subset dictionary with class (keys) and values (videos)
        class_subset - list of the keys of files_subset
        """
        subset = {k: files[k][:numvideos] for k in np.sort(list(files))[:numclass]}
        return subset, list(subset.keys())
    
    def reconstruct_data(files_subset, class_subset):
        """ 
        Transform data into a dataset with keys 'data' and 'target' (class)
        
        """
        dataset = {'data': [], 'target': []}
        for k in class_subset:
            [dataset['data'].append(file) for file in files_subset[k]] 
            [dataset['target'].append(k) for file in files_subset[k]]
        return dataset


    def generate_dataset(self, path, numclass) -> dict:
        self.path = path
        self.numclass = numclass
        
        # zip file to folder and filenames
        file_names = self.list_files_from_zip_path(self.path)
        videos = [f for f in file_names if f.endswith('MOV')]
        folders = [f for f in file_names if not f.endswith('MOV')]

        # dictionary of video per class
        complete_files = self.get_files_per_class(videos, folders)

        # subset of files
        files_subset, class_subset = self.subset_data(complete_files, self.numclass)
        
        # reconstruction of data
        dataset = self.reconstruct_data(files_subset, class_subset)
        return dataset
