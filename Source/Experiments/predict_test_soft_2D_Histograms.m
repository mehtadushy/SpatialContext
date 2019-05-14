%Using LibSVM Train
%This file too is to be run cell wise
image_dir = '../Data/SceneCategoriesDataset_Images';
data_dir = '/scratch/common/pool0/dmetha/SceneCategoriesDataset_Data/SoftAssign_2D';
data_dir2 = '/scratch/common/pool0/dmetha/SceneCategoriesDataset_Data';
load(fullfile(data_dir2, 'scenes_names.mat'));
load(fullfile(data_dir2, 'category_names.mat'));
%Do a test-train split (data is already randomized
train_percent = 90;
train.index = (1 : floor(length(data.labels)*train_percent/100));
train.labels = data.labels(train.index);
test.index = (length(train.index)+1 : length(data.labels));
test.labels = data.labels(test.index);
train_test_datafilename = fullfile(data_dir, 'train_test_data.mat');
save(train_test_datafilename, 'train', 'test');
%%
%%%%%%%%%%%%%%%%
% 2D Histograms of Annular Keypoint Neighbourhood with keypoint descriptors
%%%%%%%%%%%%%%%%
%Train SVM with intersection kernel and predict on the
%test set
sigmaFactors = [
               10 16;
               %10 18;
               6 10;
               %6 14;
               %8 16;
               ];
neighborhoodHistogramDictionarySize = [50, 100, 200, 400]; %Dense SIFT
dictionarySize = [20, 50, 80, 100, 120]; %Dic Size of Desne SIFT BoW
keypointDictionarySize = [50, 100, 200, 400]; %Keypoint SIFT
accuracy_results = cell(length(keypointDictionarySize)*size(sigmaFactors,1)+1, length(dictionarySize)*length(neighborhoodHistogramDictionarySize)+1);

for sf = 1:size(sigmaFactors,1)
   for j = 1: length(keypointDictionarySize)
       accuracy_results{(sf-1)*length(keypointDictionarySize)+j+1 ,1} = sprintf('%d_%d_Neigh  %d_KeyptDictionary', sigmaFactors(sf,1), sigmaFactors(sf,2), keypointDictionarySize(j));
     for dic = 1:length(dictionarySize)
        for k = 1: length(neighborhoodHistogramDictionarySize)
            accuracy_results{1, (dic-1)*length(neighborhoodHistogramDictionarySize)+k+1} = sprintf('%d_NeighHistDic  %d_DenseSIFTDic', dictionarySize(dic), neighborhoodHistogramDictionarySize(k));
        end
     end
   end
end
%%
sf_size = size(sigmaFactors,1);
keyptDicSize_size = length(keypointDictionarySize);
dicSize_size = length(dictionarySize);
neighHistDicSize_size = length(neighborhoodHistogramDictionarySize);
for sf = 1:sf_size
   for j = 1: keyptDicSize_size
     for dic = 1:dicSize_size
        acc_log = cell(1, neighHistDicSize_size);
        parfor k = 1: neighHistDicSize_size
            fprintf('Predicting With SF_%d_%d KeypointDic_%d HistDic_%d DenseDic_%d',sigmaFactors(sf,1), sigmaFactors(sf,2), keypointDictionarySize(j), neighborhoodHistogramDictionarySize(k),dictionarySize(dic));
            
            inFName = fullfile(data_dir, sprintf('2Dhistograms_%s_with%s.mat', sprintf('_soft_texton_ind_%d_keypoint_sift',keypointDictionarySize(j)), sprintf('_soft_texton_ind_%d_%d_%d_neighborhood_histsize%d', dictionarySize(dic), sigmaFactors(sf,1), sigmaFactors(sf,2), neighborhoodHistogramDictionarySize(k))));
            H_all = load(inFName,'-ascii');
            dat = load(train_test_datafilename);
            train = dat.train;
            test = dat.test;
            train_features = H_all(train.index,:);
            test_features = H_all(test.index,:);

            %We'll do multiclass classification
            %Compute kernel for training data
            train.K = [(1:size(train_features,1))' , intersectionKernel(train_features, train_features)];
            test.K = [(1:size(test_features,1))' , intersectionKernel(test_features, train_features)];
    
            bow_model = svmtrain(train.labels',train.K, '-t 4' );
            [predict_label_P, accuracy_P, dec_values_P] = svmpredict(test.labels', test.K, bow_model);
            acc_log{k} = accuracy_P(1);
            
%             confusion_matrix = cell(length(dirlist)+1,length(dirlist)+1);
%             for i = 1:length(dirlist)
%                 confusion_matrix{i+1,1} = dirlist{i};
%                 num_i_labels = sum(test.labels == i);
%                 for g = 1:length(dirlist)
%                     confusion_matrix{1,g+1} = dirlist{g};
%                     num_j_labels = sum(predict_label_P(test.labels == i) == g);
%                     confusion_matrix{i+1,g+1} = num_j_labels/num_i_labels;
%                 end
%             end
            
            %outfname = fullfile( data_dir, 'results',  sprintf('confusion_matrix_%s_with%s.mat', sprintf('_%d_KeyptDic',keypointDictionarySize(j)), sprintf('_%d_DenseDic_%d_%d_NeighSize_NeighHistDic%d', dictionarySize(dic), sigmaFactors(sf,1), sigmaFactors(sf,2), neighborhoodHistogramDictionarySize(k))));
            %save(outfname, 'confusion_matrix');
            
        end
        accuracy_results((sf-1)*keyptDicSize_size+j+1 ,(dic-1)*neighHistDicSize_size+2:(dic)*neighHistDicSize_size+1) = acc_log;
      end
    end
end

outfname = fullfile( data_dir, 'results', '2DHist_accuracy.mat');
save(outfname, 'accuracy_results');







