% Read scene_categories data and create some data structure that would be
% used later on. This can be used for other datasets as well
image_dir = '~/DirectionalPooling/directionalpooling/Data/SceneCategoriesDataset_Images';
data_dir = '/scratch/common/pool0/dmetha/SceneCategoriesDataset_Data';
curr_dir = dirPoolCodePath; % We need to come back to this directory
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

data.filenames = filenames(rand_perm);
data.labels = labels(rand_perm);
cd (curr_dir);  %going back to the directory we were in
save(fullfile(data_dir, 'scenes_names.mat'),'data');
%% This serves to create a list of the category names to be used later in confusion matrices and such
for i = 1:length(dirlist)
    dirlist{i} = dirlist{i}(3:end);
end
save(fullfile(data_dir, 'category_names.mat'), 'dirlist');