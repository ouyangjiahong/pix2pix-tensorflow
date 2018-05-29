clear
num = 0;
% load annotation_list.mat
ann_dir = '/home/jihang/Jiahong/data/OCT/Annotations/';
miss_path = '/home/jihang/Jiahong/data/OCT/miss/';

margin = 0;
% for t = [1,22,25,26,28,32,33,34,3,4,6,8,13,14,21]    %dataset
for t = [15, "corneaSubject1", "corneaSubject5", "corneaSubject6", "corneaSubject7", ...
        "corneaSubject8", "palisades25_102AM_040417", "palisades28_172_180213",...
        "palisades31_187_180213"]
    t = char(t)
    ann_path = [ann_dir num2str(t) '/originalAnnotationTejas_new/'];
    ann_directory = dir(ann_path);
    miss_path_tmp = [miss_path t]
    mkdir(miss_path_tmp)
    for i = 1 : length(ann_directory)
        if(isequal(ann_directory(i).name, '.') || isequal(ann_directory(i).name, '..'))   %exclude hidden folder & not folder
            continue;
        end
        img_ann_path = [ann_path ann_directory(i).name]
        img_ann = imread(img_ann_path);
        [row col cha] = size(img_ann);
        
        flag = 0;
        misscol = 0;
        for j = 1 : col
            k = 1;
            while(k <= row - 5)
                if(img_ann(k+margin,j,1)>=200 && img_ann(k+margin,j,2)<200 && img_ann(k+margin,j,3)<200)
                    break;
                end
                k = k + 1;
            end
            if(k > row - 5)
                if flag == 0
                    flag = 1;
                    num = num + 1;
                    miss_path_list{num,1} = img_ann_path;
                    img_miss_path = [miss_path num2str(t) '/' ann_directory(i).name]
                    copyfile(img_ann_path, img_miss_path);
                end
                misscol = misscol + 1;
                tmp = j
%                 break;
            end
        end
        if flag == 1
            miss_path_list{num,2} = misscol;
        end
    end
end