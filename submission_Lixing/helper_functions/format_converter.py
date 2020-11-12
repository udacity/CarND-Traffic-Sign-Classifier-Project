from PIL import Image as img

def img_2jpg(path, name, new_path="./"):
    try:
        img1 = img.open(path + name)    # read the data
    except:
        print("Unable to open the image!")
        return
    img1.show()     # Preview the data

    name_list = name.split('.')    # Convert to jpg
    name_new = ""
    for sub_name in name_list[:-1]:
        name_new = name_new + sub_name + '.'
    name_new += "jpg"
    img1.save(new_path + name_new)

if __name__ == "__main__":
    #img_2jpg("./submission_Lixing/datasets/training/Images/00005/", "00000_00012.ppm", "./submission_Lixing/references/")

    for i in range(43):
        file_path = "../references/classes/"
        file_name = "c" + str(i) + ".ppm"
        img_2jpg(file_path, file_name, file_path)