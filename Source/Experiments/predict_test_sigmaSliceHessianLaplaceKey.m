%Testing Sigma Slices computed around HessianLaplace keypoints
%Using LibSVM Train
%This file too is to be run cell wise
image_dir = '..\..\Data\SceneCategoriesDataset_Images';
data_dir = '..\..\Data\SceneCategoriesDataset_Data';

%load the labels and the filepaths
load(fullfile(data_dir, 'scenes_names.mat'));
%Do a test-train split (data is already randomized
train_percent = 90;
train.index = (1 : floor(length(data.labels)*train_percent/100));
train.labels = data.labels(train.index);
test.index = (length(train.index)+1 : length(data.labels));
test.labels = data.labels(test.index);
%%
%%%%%%%%%%%%%%%%
% Sigma Slices
%%%%%%%%%%%%%%%%
%Train SVM with intersection kernel for Sigma Slices and predict on the
%test set
params.keypointDetector = 'DoG';
dictionarySize = [400]; %[50, 100, 200, 400];
sigmaFactors = [3 7 8];%[2 4 6; 2 4 8; 2 4 10; 1 4 6; 1 4 8; 1 4 10; 3 6 8; 3 7 8];
numKeypoints = [30];%[10 20 30];
sigmaslice_accuracy_all = cell(length(dictionarySize),2);
for i = 1: length(dictionarySize)
    sigmaslice_accuracy = cell(length(numKeypoints)+1, size(sigmaFactors,1)+1);
    for j = 1: length(numKeypoints)
        sigmaslice_accuracy{j+1,1} = sprintf('%d Keypoints',numKeypoints(j));
        for k = 1:size(sigmaFactors,1)
            params.dictionarySize = dictionarySize(i);
            params.sigmaFactors = sigmaFactors(k,:); 
            params.sigmaKeypoints = numKeypoints(j);
            sigmaslice_accuracy{1,k+1} = sprintf('%d_%d_%d Sigma Factors',params.sigmaFactors(1),params.sigmaFactors(2),params.sigmaFactors(3));
            
            inFName = fullfile(data_dir, sprintf('sigma_slice_%d_dictionary_%d_%s_keypoints_%d_%d_%d_scale.mat',params.dictionarySize, params.sigmaKeypoints, params.keypointDetector, params.sigmaFactors(1),params.sigmaFactors(2),params.sigmaFactors(3)));
            load(inFName,'sigmaSlices');
            sigmaSlices = [0.25*sigmaSlices(:,1:params.dictionarySize) 0.5*sigmaSlices(:,params.dictionarySize+1:2*params.dictionarySize) 0.25*sigmaSlices(:,2*params.dictionarySize+1:3*params.dictionarySize)];
            train.features = sigmaSlices(train.index,:);
            test.features = sigmaSlices(test.index,:);

            %We'll do multiclass classification
            %Compute kernel for training data
            train.K = [(1:size(train.features,1))' , intersectionKernel(train.features, train.features)];
            test.K = [(1:size(test.features,1))' , intersectionKernel(test.features, train.features)];
    
            bow_model = svmtrain(train.labels',train.K, '-t 4' );
            [predict_label_P, accuracy_P, dec_values_P] = svmpredict(test.labels', test.K, bow_model);
            sigmaslice_accuracy{j+1,k+1}= accuracy_P(1);
        end
    end
    sigmaslice_accuracy_all{i,1} = dictionarySize(i);
    sigmaslice_accuracy_all{i,2} = sigmaslice_accuracy;
    outfname = fullfile( data_dir, sprintf('sigmaslice_%d_dictionary_%s_accuracy.mat',dictionarySize(i), params.keypointDetector) );
    save(outfname, 'sigmaslice_accuracy');
end
outfname = fullfile( data_dir, 'sigmaslice_accuracy_DoG.mat');
save(outfname, 'sigmaslice_accuracy_all');


%%
%%%%%%%%%%%%%%%%
% Histogram augmented annular Sigma Slices
%%%%%%%%%%%%%%%%
%Train SVM with intersection kernel for Sigma Slices and predict on the
%test set
params.keypointDetector = 'HessianLaplace';
dictionarySize = [400]; %[50, 100, 200, 400];
sigmaFactors = [4 8 12];%[2 4 6; 2 4 8; 2 4 10; 1 4 6; 1 4 8; 1 4 10; 3 6 8; 3 7 8];
numKeypoints = [30];%[10 20 30];
sigmaslice_accuracy_all = cell(length(dictionarySize),2);
for i = 1: length(dictionarySize)
    sigmaslice_accuracy = cell(length(numKeypoints)+1, size(sigmaFactors,1)+1);
    histogram = load(fullfile(data_dir,sprintf('histograms_%d.mat',dictionarySize(i))),'-ascii');
    for j = 1: length(numKeypoints)
        sigmaslice_accuracy{j+1,1} = sprintf('%d Keypoints',numKeypoints(j));
        for k = 1:size(sigmaFactors,1)
            params.dictionarySize = dictionarySize(i);
            params.sigmaFactors = sigmaFactors(k,:); 
            params.sigmaKeypoints = numKeypoints(j);
            sigmaslice_accuracy{1,k+1} = sprintf('%d_%d_%d Sigma Factors',params.sigmaFactors(1),params.sigmaFactors(2),params.sigmaFactors(3));
            
            inFName = fullfile(data_dir, sprintf('sigma_slice_%d_dictionary_%d_%s_keypoints_%d_%d_%d_scale.mat',params.dictionarySize, params.sigmaKeypoints, params.keypointDetector, params.sigmaFactors(1),params.sigmaFactors(2),params.sigmaFactors(3)));
            load(inFName,'sigmaSlices');
            %feature = [0.25*sigmaSlices(:,1:params.dictionarySize) 0.25*(sigmaSlices(:,params.dictionarySize+1:2*params.dictionarySize)-sigmaSlices(:,1:params.dictionarySize)) 0.1*(sigmaSlices(:,2*params.dictionarySize+1:3*params.dictionarySize)-sigmaSlices(:,params.dictionarySize+1:2*params.dictionarySize)) histogram];
            feature = [histogram];
            train.features = feature(train.index,:);
            test.features = feature(test.index,:);

            %We'll do multiclass classification
            %Compute kernel for training data
            train.K = [(1:size(train.features,1))' , intersectionKernel(train.features, train.features)];
            test.K = [(1:size(test.features,1))' , intersectionKernel(test.features, train.features)];
    
            bow_model = svmtrain(train.labels',train.K, '-t 4 -c 0.8' );
            [predict_label_P, accuracy_P, dec_values_P] = svmpredict(test.labels', test.K, bow_model);
            sigmaslice_accuracy{j+1,k+1}= accuracy_P(1);
        end
    end
    sigmaslice_accuracy_all{i,1} = dictionarySize(i);
    sigmaslice_accuracy_all{i,2} = sigmaslice_accuracy;
    outfname = fullfile( data_dir, sprintf('sigmaslice_%d_dictionary_%s_accuracy_aughist.mat',dictionarySize(i), params.keypointDetector) );
    save(outfname, 'sigmaslice_accuracy');
end
outfname = fullfile( data_dir, 'sigmaslice_accuracy_HessianLaplace_aughist.mat');
save(outfname, 'sigmaslice_accuracy_all');

%%
%Create confusion matrix from the predicted label

curr_dir = pwd; % We need to come back to this directory
cd (image_dir);
dirlist = textscan(genpath('.'),'%s','delimiter',';');
dirlist = dirlist{1};
dirlist = dirlist(2:end); %These are subdirectory names, to be added to the files that are found
cd (curr_dir);
save(fullfile(data_dir, 'category_names.mat'), 'dirlist');
%%
%load(fullfile(data_dir, 'category_names.mat'), 'dirlist');
confusion_matrix = cell(length(dirlist)+1,length(dirlist)+1);
for i = 1:length(dirlist)
    confusion_matrix{i+1,1} = dirlist{i};
    num_i_labels = sum(test.labels == i);
    for j = 1:length(dirlist)
        confusion_matrix{1,j+1} = dirlist{j};
        num_j_labels = sum(predict_label_P(test.labels == i) == j);
        confusion_matrix{i+1,j+1} = num_j_labels/num_i_labels;
    end
end
        

