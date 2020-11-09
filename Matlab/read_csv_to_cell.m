function datacell = read_csv_to_cell(fname)
%% datacell = read_csv_to_cell(fname) reads a csv file into a cell array, 
% where each cell element is a matrix. The csv file is assumed to have a
% header and the last column indicating the class labels.
% 
% Parameters
% ----------
%
% fname : string
%         Name of the csv file to be read.
%
% datacell : cell of size (n_classes, 1)
%            Each element of the cell contains the data from a particular
%            class in the form of a matrix of size (n_samples, n_features).
%
% Example of csv file format supported by the function.
%
%0,1,2,class
%-0.5813760371451253,0.028594642962316397,0.2995958799693469,0
%3.156689532751944,0.22533160227332705,0.6672443949708514,0
%0.9356741338649087,0.2414127768461044,0.618401331366264,1
%0.01167725340779635,-0.6377146053889979,1.1017071550787232,1



T = readtable(fname);

data = T;
if istable(data)
    data = table2array(data);
    X = data(2:end,1:end-1);
    y = data(2:end,end);
end

classes = unique(y);
K = length(classes);
datacell = cell(K,1);
for k=1:K
    datacell{k} = X(y==classes(k),:);
end
    