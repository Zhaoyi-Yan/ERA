"""
You should generate the `train.txt` and `val_tmp.txt` first.
The train.txt should be like this:
`folder_name/image_name  category`

For example:
```
n01440764/n01440764_10026.JPEG 0
n01440764/n01440764_10027.JPEG 0
...
```

The `val_tmp.txt` should be:
```
n01751748/ILSVRC2012_val_00000001.JPEG
```

Then running this script, the script will assigning the corresponding category to each image.
The generated `val.txt` should be:
```
n01751748/ILSVRC2012_val_00000001.JPEG 65
n09193705/ILSVRC2012_val_00000002.JPEG 970
n02105855/ILSVRC2012_val_00000003.JPEG 230
...
```

"""


# create a dictionary mapping cls_num to folder_cls
folder_cls_dict = {}

with open('train.txt', 'r') as file:
    for line in file:
        folder_cls, cls_num = line.strip().split()
        cls_num = int(cls_num)
        if cls_num not in folder_cls_dict:
            folder_cls = folder_cls.split('/')[0]
            folder_cls_dict[cls_num] = folder_cls

# assigning the folder category to the val image path
with open('val_tmp.txt', 'r') as file, open('val.txt', 'w') as output_file:
    for line in file:
        img_name, cls_num = line.strip().split()
        cls_num = int(cls_num)
        if cls_num in folder_cls_dict:
            new_line = f"{folder_cls_dict[cls_num]}/{img_name} {cls_num}\n"
            output_file.write(new_line)
        else:
            print(f"Class number: {cls_num} not found in train labels.")

