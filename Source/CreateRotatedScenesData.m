% Read scene_categories data and create some data structure that would be
% used later on. This can be used for other datasets as well. Also creates 
%image_dir = '~/DirectionalPooling/directionalpooling/Data/SceneCategoriesDataset_RotatedImages';
image_dir = '~/DirectionalPooling/directionalpooling/Data/SceneCategoriesDataset_RotatedImages';
data_dir = '/scratch/common/pool0/dmetha/SceneCategoriesDataset_Data';
curr_dir = pwd; % We need to come back to this directory
cd (image_dir);

dirlist = textscan(genpath('.'),'%s','delimiter',':');

dirlist = dirlist{1};
dirlist = dirlist(2:end); %These are subdirectory names, to be added to the files that are found
filenames = cell(1,1);
labels = [];
f_idx = 1;
for i = 1:length(dirlist)
    %for each directory, get filenames and append to 
    fnames = dir(fullfile(dirlist{i}, '*.jpg'));
    num_files = size(fnames,1);
    for f = 1:num_files
	  filenames{f+f_idx-1} = fullfile(dirlist{i}(3:end), fnames(f).name);
      labels(f+f_idx-1) = i;
    end
    f_idx = f_idx + num_files;
end
    
rand_perm = randperm(length(filenames));
%Pick some % of files to create rotated images for, rotated at random
%between -90 and 90
filenames = filenames(rand_perm);
labels = labels(rand_perm);
rot_filenames = cell(1,1);
rot_labels = [];
rot_angles = [180 165 90 75 30 -30 -75 -90];
for f = 1:0.75*length(filenames)
     rot = rot_angles(randi(8,1,1));
     imageFName = filenames{f};
     [dirN base] = fileparts(imageFName);
     baseFName = [dirN filesep base];
     outFName = fullfile(image_dir, sprintf('%s_rot.jpg', baseFName));
     inFName = fullfile(image_dir, filenames{f});
     img = imread(inFName);
     img = imrotate(img, rot, 'bilinear');
     imwrite(img, outFName);
     rot_filenames{f} = sprintf('%s_rot.jpg', baseFName);
     rot_labels(f) = labels(f);
end
filenames = [filenames, rot_filenames];
labels = [labels, rot_labels];
rand_perm = randperm(length(filenames));
data.filenames = filenames(rand_perm);
data.labels = labels(rand_perm);
cd (curr_dir);  %going back to the directory we were in
save(fullfile(data_dir, 'rot_scenes_names.mat'),'data');
%% This serves to create a list of the category names to be used later in confusion matrices and such
for i = 1:length(dirlist)
    dirlist{i} = dirlist{i}(3:end);
end
save(fullfile(data_dir, 'rot_category_names.mat'), 'dirlist');
