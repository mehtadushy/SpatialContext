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
dictionary_size = [50, 100, 200, 400];
pyramid_levels = [2, 3, 4];

%%

%Train SVM with intersection kernel for BoW, and predict on the test set
%It is multi-class classification, done one vs one, and same parameters
% used for all discriminants
%Cross validation ??

bow_accuracy = zeros(size(dictionary_size));
for i = 1:length(dictionary_size)
    histogram = load(fullfile(data_dir,sprintf('histograms_%d.mat',dictionary_size(i))),'-ascii');
    train.features = histogram(train.index,:);
    test.features = histogram(test.index,:);

    %We'll do multiclass classification
    %Compute kernel for training data
    train.K = [(1:size(train.features,1))' , intersectionKernel(train.features, train.features)];
    test.K = [(1:size(test.features,1))' , intersectionKernel(test.features, train.features)];
    
    bow_model = svmtrain(train.labels',train.K, '-t 4' );
    [predict_label_P, accuracy_P, dec_values_P] = svmpredict(test.labels', test.K, bow_model);
    bow_accuracy(i) = accuracy_P(1);
end

outfname = fullfile( data_dir, 'bow_accuracy.mat');
save(outfname, 'bow_accuracy', 'dictionary_size');
%%
%Train SVM with intersection kernel for Pyramid Kernels, and predict on the test set
%It is multi-class classification, done one vs one, and same parameters
% used for all discriminants

pyramid_accuracy = zeros( length(dictionary_size)+1 , length(pyramid_levels)+1);
pyramid_accuracy(2:end,1) = dictionary_size';
pyramid_accuracy(1, 2:end) = pyramid_levels;

for j = 1:length(pyramid_levels)
    for i = 1:length(dictionary_size)
        fprintf('Level %d Dictionary Size %d\n', pyramid_levels(j), dictionary_size(i));
        load(fullfile(data_dir,sprintf('pyramids_all_%d_%d.mat',dictionary_size(i), pyramid_levels(j))));
        train.features = pyramid_all(train.index,:);
        test.features = pyramid_all(test.index,:);

        %We'll do multiclass classification
        %Compute kernel for training data
        train.K = [(1:size(train.features,1))' , intersectionKernel(train.features, train.features)];
        test.K = [(1:size(test.features,1))' , intersectionKernel(test.features, train.features)];
    
        bow_model = svmtrain(train.labels',train.K, '-t 4' );
        [predict_label_P, accuracy_P, dec_values_P] = svmpredict(test.labels', test.K, bow_model);
        pyramid_accuracy(i+1,j+1) = accuracy_P(1);
    end
    accuracy = pyramid_accuracy(2:end, j+1);
    outfname = fullfile( data_dir, sprintf('pyramid_%d_accuracy.mat',pyramid_levels(j)) );
    save(outfname, 'accuracy');
end

outfname = fullfile( data_dir, 'pyramid_accuracy.mat');
save(outfname, 'pyramid_accuracy');


%%
%%%%%%%%%%%%%%%%
% Sigma Slices
%%%%%%%%%%%%%%%%
%Train SVM with intersection kernel for Sigma Slices and predict on the
%test set
dictionarySize = [50, 100, 200, 400];
sigmaFactors = [2 4 6; 2 4 8; 2 4 10; 1 4 6; 1 4 8; 1 4 10; 3 6 8; 3 7 8];
numKeypoints = [10 20 30];
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
            
            inFName = fullfile(data_dir, sprintf('sigma_slice_%d_dictionary_%d_keypoints_%d_%d_%d_scale.mat',params.dictionarySize, params.sigmaKeypoints, params.sigmaFactors(1),params.sigmaFactors(2),params.sigmaFactors(3)));
            load(inFName,'sigmaSlices');
            sigmaSlices = [0.5*sigmaSlices(:,1:params.dictionarySize) 0.5*sigmaSlices(:,params.dictionarySize+1:2*params.dictionarySize) 0.25*sigmaSlices(:,2*params.dictionarySize+1:3*params.dictionarySize)];
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
    outfname = fullfile( data_dir, sprintf('sigmaslice_%d_dictionary_accuracy.mat',dictionarySize(i)) );
    save(outfname, 'sigmaslice_accuracy');
end
outfname = fullfile( data_dir, 'sigmaslice_accuracy.mat');
save(outfname, 'sigmaslice_accuracy_all');

%%
%%%%%%%%%%%%%%%%
% Sigma Slices with different weighing
%%%%%%%%%%%%%%%%
%Train SVM with intersection kernel for Sigma Slices and predict on the
%test set
dictionarySize = [200, 400];
sigmaFactors = [2 4 10; 3 7 8; 3 7 9];
numKeypoints = [30];
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
            
            inFName = fullfile(data_dir, sprintf('sigma_slice_%d_dictionary_%d_keypoints_%d_%d_%d_scale.mat',params.dictionarySize, params.sigmaKeypoints, params.sigmaFactors(1),params.sigmaFactors(2),params.sigmaFactors(3)));
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
    outfname = fullfile( data_dir, sprintf('sigmaslice_%d_dictionary_accuracy_diff_wght.mat',dictionarySize(i)) );
    save(outfname, 'sigmaslice_accuracy');
end
outfname = fullfile( data_dir, 'sigmaslice_accuracy_diff_wght.mat');
save(outfname, 'sigmaslice_accuracy_all');

%%
%%%%%%%%%%%%%%%%
% Histogram augmented annular Sigma Slices
%%%%%%%%%%%%%%%%
%Train SVM with intersection kernel for Sigma Slices and predict on the
%test set
dictionarySize = [200, 400];
sigmaFactors = [2 4 6; 2 4 8; 2 4 10; 1 4 8; 3 6 8; 3 7 8];
numKeypoints = [20 30];
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
            
            inFName = fullfile(data_dir, sprintf('sigma_slice_%d_dictionary_%d_keypoints_%d_%d_%d_scale.mat',params.dictionarySize, params.sigmaKeypoints, params.sigmaFactors(1),params.sigmaFactors(2),params.sigmaFactors(3)));
            load(inFName,'sigmaSlices');
            feature = [0.25*sigmaSlices(:,1:params.dictionarySize) 0.5*(sigmaSlices(:,params.dictionarySize+1:2*params.dictionarySize)-sigmaSlices(:,1:params.dictionarySize)) 0.25*(sigmaSlices(:,2*params.dictionarySize+1:3*params.dictionarySize)-sigmaSlices(:,params.dictionarySize+1:2*params.dictionarySize)) 0.5*histogram];
            %feature = [sigmaSlices histogram];
            train.features = feature(train.index,:);
            test.features = feature(test.index,:);

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
    outfname = fullfile( data_dir, sprintf('sigmaslice_%d_dictionary_accuracy_aughist.mat',dictionarySize(i)) );
    save(outfname, 'sigmaslice_accuracy');
end
outfname = fullfile( data_dir, 'sigmaslice_accuracy_aughist.mat');
save(outfname, 'sigmaslice_accuracy_all');

%%
%%%%%%%%%%%%%%%%
% Histogram augmented immediate neighbourhood pooling
%%%%%%%%%%%%%%%%
%Train SVM with intersection kernel for Sigma Slices and predict on the
%test set
dictionarySize = [200, 400];
sigmaFactors = [3 6 8];
numKeypoints = [10 20 30];
sigmaslice_accuracy_all = cell(length(dictionarySize),2);
for i = 1: length(dictionarySize)
    sigmaslice_accuracy = cell(length(numKeypoints)+1, length(sigmaFactors)+1);
    histogram = load(fullfile(data_dir,sprintf('histograms_%d.mat',dictionarySize(i))),'-ascii');
    for j = 1: length(numKeypoints)
        sigmaslice_accuracy{j+1,1} = sprintf('%d Keypoints',numKeypoints(j));
        for k = 1:length(sigmaFactors)
            params.dictionarySize = dictionarySize(i);
            params.sigmaFactors = sigmaFactors; 
            params.sigmaKeypoints = numKeypoints(j);
            sigmaslice_accuracy{1,k+1} = sprintf('%d_ Sigma Factor with hist',params.sigmaFactors(k));
            
            inFName = fullfile(data_dir, sprintf('sigma_slice_%d_dictionary_%d_keypoints_%d_%d_%d_scale.mat',params.dictionarySize, params.sigmaKeypoints, params.sigmaFactors(1),params.sigmaFactors(2),params.sigmaFactors(3)));
            load(inFName,'sigmaSlices');
            feature = [0.5*sigmaSlices(:,(k-1)*params.dictionarySize+1:k*params.dictionarySize)  0.25*histogram];
            %feature = [sigmaSlices histogram];
            train.features = feature(train.index,:);
            test.features = feature(test.index,:);

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
    outfname = fullfile( data_dir, sprintf('sigmaslice_%d_dictionary_accuracy_aughist_immediate.mat',dictionarySize(i)) );
    save(outfname, 'sigmaslice_accuracy');
end
outfname = fullfile( data_dir, 'sigmaslice_accuracy_aughist_immediate.mat');
save(outfname, 'sigmaslice_accuracy_all');
