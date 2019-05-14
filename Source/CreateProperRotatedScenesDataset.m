% Read scene_categories data and create some data structure that would be
% used later on. This can be used for other datasets as well.
% Train and test sets are kept separate, ie, first 90% of the images and
% their rotated copies are kept separate from the other 10% and its rotated
% copies
%image_dir = '~/DirectionalPooling/directionalpooling/Data/SceneCategoriesDataset_RotatedImages';
clear;
image_dir = '~/DirectionalPooling/directionalpooling/Data/SceneCategoriesDataset_RotImages3';
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

train_set_length = floor(0.9 * length(filenames));
train_filenames = filenames(1:train_set_length);
test_filenames = filenames(train_set_length+1:end);

train_labels = labels(1:train_set_length);
test_labels = labels(train_set_length+1:end);


rot_angles = [180 165 90 75 30 -30 -75 -90];
%Create rotated copies of test set
rot_filenames = cell(1,1);
rot_labels = [];
for f = 1:length(test_filenames)
     rot = rot_angles(randi(8,1,1));
     imageFName = test_filenames{f};
     [dirN base] = fileparts(imageFName);
     baseFName = [dirN filesep base];
     outFName = fullfile(image_dir, sprintf('%s_rot.jpg', baseFName));
     inFName = fullfile(image_dir, test_filenames{f});
     img = imread(inFName);
     img = imrotate(img, rot, 'bilinear');
     imwrite(img, outFName);
     rot_filenames{f} = sprintf('%s_rot.jpg', baseFName);
     rot_labels(f) = test_labels(f);
end
test_filenames = [test_filenames, rot_filenames];
test_labels = [test_labels, rot_labels];

%Create rotated copies of 10% of the training set
rot_filenames = cell(1,1);
rot_labels = [];
for f = 1:0.10*length(train_filenames)
     rot = rot_angles(randi(8,1,1));
     imageFName = train_filenames{f};
     [dirN base] = fileparts(imageFName);
     baseFName = [dirN filesep base];
     outFName = fullfile(image_dir, sprintf('%s_rot.jpg', baseFName));
     inFName = fullfile(image_dir, train_filenames{f});
     img = imread(inFName);
     img = imrotate(img, rot, 'bilinear');
     imwrite(img, outFName);
     rot_filenames{f} = sprintf('%s_rot.jpg', baseFName);
     rot_labels(f) = train_labels(f);
end
test_filenames = [test_filenames, rot_filenames];
test_labels = [test_labels, rot_labels];

rand_perm = randperm(length(test_filenames));
test_filenames = test_filenames(rand_perm);
test_labels = test_labels(rand_perm);


data.filenames = [train_filenames, test_filenames];
data.labels = [train_labels, test_labels];
train.index = 1:length(train_filenames);
test.index = length(train_filenames)+1:length(data.filenames);
train.labels = train_labels;
test.labels = test_labels;


save(fullfile(data_dir, 'rot3_scenes_names.mat'),'data','train', 'test');
%% This serves to create a list of the category names to be used later in confusion matrices and such
dirlist = textscan(genpath('.'),'%s','delimiter',':');

dirlist = dirlist{1};
dirlist = dirlist(2:end); %These are subdirectory names, to be added to the files that are found
for i = 1:length(dirlist)
    dirlist{i} = dirlist{i}(3:end);
end
save(fullfile(data_dir, 'rot3_category_names.mat'), 'dirlist');

cd (curr_dir);  %going back to the directory we were in
