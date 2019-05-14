% Computes a stronger(than just relative orientation) context that captures
% both relative orientation and types of SIFTs occurring in the
% neighbourhood of a keypoint through a 2D histogram. The relative
% orientation is pushed into the keypoint orientation, while a coarse
% sift histogram from the neighbourhood is clustered orthogonally and 
% and VLAD encoded. SVM then needs to use inner product instead of histogram
% intersection

clear;
image_dir = '~/DirectionalPooling/directionalpooling/Data/SceneCategoriesDataset_Images';
data_dir = '/scratch/common/pool0/dmetha/SceneCategoriesDataset_Data';
data_dir2 = '/scratch/common/pool0/dmetha/SceneCategoriesDataset_Data';
load(fullfile(data_dir2, 'scenes_names.mat'));

coarseDictionarySize = 40; %[20, 40, 50]; %Coarse Dictionary Size


%% Next up, we would construct VLAD encoded 3D Histograms 
neighbourhoodSize = [ 40;
                       80;
                       ];
%coarseDictionarySize = [20, 40, 50]; %Coarse Dictionary Size
contextDictionarySize = [10 ,20, 40];
keypointDictionarySize = [50, 100, 200, 400,800, 1000];
config_table = zeros(length(keypointDictionarySize) * length(coarseDictionarySize) * length(contextDictionarySize) * length(neighbourhoodSize), 4);
idx = 1;
for j = 1: length(keypointDictionarySize)
    for l = 1: length(coarseDictionarySize)
     for k = 1: length(contextDictionarySize)
      for i = 1:length(neighbourhoodSize)
      config_table(idx,:) = [keypointDictionarySize(j), coarseDictionarySize(l), contextDictionarySize(k), neighbourhoodSize(i)];
      idx= idx+1;
      end
    end
  end
end
parfor idx = 1: size(config_table,1)
            pfig = sp_progress_bar(sprintf('2D Hist of Neigh Size%d Keypt Dic%d Context Dic%d',config_table(idx,4), config_table(idx,1), config_table(idx,3)));
            CompileVLADTemp( data.filenames,data_dir, sprintf('_soft_texton_ind_%d_%d_neigh_ROcontext_keypoint_sift',config_table(idx,1),config_table(idx,4)), sprintf('_soft_texton_ind_%d_%d_neigh_contexthist%d', config_table(idx,3), config_table(idx,4), config_table(idx,2)), [config_table(idx,1), config_table(idx,3), config_table(idx,2)],[] , 0, pfig );
            close(pfig)
end
%%
%


