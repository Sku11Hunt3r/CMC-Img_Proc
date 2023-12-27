import argparse  # модуль (библиотека) для обработки параметров коммандной строки
import numpy as np  # модуль для работы с массивами и векторных вычислений
import skimage.io  # модуль для обработки изображений, подмодуль для чтения и записи
# в некоторых модулях некоторые подмодули надо импортировать вручную, а не просто "import module" и потом в коде писать "module.submodule.something..."


def mirror(img, axis='h'):  # можно задавать значения параметров по умолчанию
    '''Отразить изображение'''  # комментарий docsting - выводится в подсказке к функции
    height, width = img.shape[:2]  # 3-я ось (каналы цветов) нам здесь не нужна
    if axis in ['h', 'v']:
        new_shape = (height, width)  # создать tuple из 2 элементов
    else:
        new_shape = (width, height)
        
    res = np.zeros(new_shape, dtype=float)  # массив из нулей
    
    if(axis == 'h'):
        for i in range(height):
            for j in range(width):
                res[i, j] = img[height - i - 1, j]
    elif (axis == 'v'):
        for i in range(height):
            for j in range(width):
                res[i, j] = img[i, width - j - 1]
    elif (axis == 'd'):
        for i in range(height):
            for j in range(width):
                res[j, i] = img[i, j]
    elif (axis == 'cd'):
        for i in range(height):
            for j in range(width):
                res[j, i] = img[height - i - 1, width - j - 1]
    
    return res


def extract(img, left_x, top_y, width, height: int):  # можно задавать типы параметров: будут выводится в подсказке к функции, но проверки типов нет
    res = np.empty((height, width), dtype=float)  # просто массив, без заполнения нулями или единицами после создания
    
    tmp_x, tmp_y = left_x, top_y
    for i in range(height):
        for j in range(width):
            if (tmp_x < 0 or tmp_x >= len(img[0]) or tmp_y < 0 or tmp_y >= len(img)):
                res[i, j] = 0
            else:
                res[i, j] = img[tmp_y, tmp_x]
            tmp_x += 1
        tmp_x = left_x
        tmp_y += 1
                
    return res


def rotate(img, direction, angle):
    height = len(img)
    width = len(img[0])
    if((angle // 90) % 4 == 0):
        return img
    elif((angle // 90) % 4 == 2):
        new_shape = (height, width)
        res = np.zeros(new_shape, dtype=float)
        for i in range(height):
            for j in range(width):
                res[i, j] = img[height - i - 1, width - j - 1]
    else:
        new_shape = (width, height)
        res = np.zeros(new_shape, dtype=float)
        if(direction == 'cw'):
            if((angle // 90) % 4 == 1):
                for i in range(height):
                    for j in range(width):
                        res[j, i] = img[height - i - 1, j]
            elif((angle // 90) % 4 == 3):
                for i in range(height):
                    for j in range(width):
                        res[j, i] = img[i, width - j - 1]
        elif(direction == 'ccw'):
            if((angle // 90) % 4 == 1):
                for i in range(height):
                    for j in range(width):
                        res[j, i] = img[i, width - j - 1]
            elif((angle // 90) % 4 == 3):
                for i in range(height):
                    for j in range(width):
                        res[j, i] = img[height - i - 1, j]
                
    return res


def autocontrast(img):
    res = np.zeros_like(img)  # массив из нулей такой же формы и типа
    
    max_value = 0
    min_value = 255
    height = len(img)
    width = len(img[0])
    for i in range(height):
        for j in range(width):
            if(img[i, j] < min_value):
                min_value = img[i, j]
            if(img[i, j] > max_value):
                max_value = img[i, j]
    for i in range(height):
        for j in range(width):
            res[i, j] = (img[i, j] - min_value) / (max_value - min_value)
                
    return res


def fixinterlace(img):
    res = img.copy()  # копия массива - конечно, тут это не надо, но для демонстрации возможностей
    
    height = len(img)
    width = len(img[0])
    for i in range(height // 2):
        for j in range(width):
            tmp =  res[2*i, j]
            res[2*i, j] =  res[2*i + 1, j]
            res[2*i + 1, j] = tmp

    def calc_variation(img):  # эту функцию можно вынести наружу к остальным функциям, а можно объявить и внутри этой
        var = 0
        height = len(img)
        width = len(img[0])
        for i in range(height - 1):
            for j in range(width):
                var += abs(img[i + 1, j] - img[i, j]) 
        return var
    
    if(calc_variation(img) < calc_variation(res)):
        return img 
    return res


if __name__ == '__main__':  # если файл выполняется как отдельный скрипт (python script.py), то здесь будет True. Если импортируется как модуль, то False. Без этой строки весь код ниже будет выполняться и при импорте файла в виде модуля (например, если захотим использовать эти функции в другой программе), а это не всегда надо.
    # получить значения параметров командной строки
    parser = argparse.ArgumentParser(  # не все параметры этого класса могут быть нужны; читайте мануалы на docs.python.org, если интересно
        prog='ProgramName',
        description='What the program does',
        epilog='Text at the bottom of help',  # в конце списка параметров и при создании list, tuple, dict и set можно оставлять запятую, чтобы можно было удобно комментить или добавлять новые строчки без добавления и удаления новых запятых
    )
    parser.add_argument('command', help='Command description')  # add_argument() поддерживает параметры вида "-p 0.1", может сохранять их как числа, строки, включать/выключать переменные True/False ("--activate-someting"), поддерживает задачу значений по умолчанию; полезные параметры: action, default, dest - изучайте, если интересно
    parser.add_argument('parameters', nargs='*')  # все параметры сохранятся в список: [par1, par2,...] (или в пустой список [], если их нет)
    parser.add_argument('input_file')
    parser.add_argument('output_file')
    args = parser.parse_args()

    # Можете посмотреть, как распознаются разные параметры. Но в самом решении лишнего вывода быть не должно.
    # print('Распознанные параметры:')
    # print('Команда:', args.command)  # между 2 выводами появится пробел
    # print('Её параметры:', args.parameters)
    # print('Входной файл:', args.input_file)
    # print('Выходной файл:', args.output_file)

    img = skimage.io.imread(args.input_file)  # прочитать изображение
    img = img / 255  # перевести во float и диапазон [0, 1]
    if len(img.shape) == 3:  # оставим только 1 канал (пусть будет 0-й) для удобства: всё равно это ч/б изображение
        img = img[:, :, 0]

    # получить результат обработки для разных комманд
    if args.command == 'mirror':
        res = mirror(img, args.parameters[0])

    elif args.command == 'extract':
        left_x, top_y, width, height = [int(x) for x in args.parameters]  # создать список из сконвертированных параметров и разложить по 4 переменным
        res = extract(img, left_x, top_y, width, height)

    elif args.command == 'rotate':
        direction = args.parameters[0]
        angle = int(args.parameters[1])
        res = rotate(img, direction, angle)

    elif args.command == 'autocontrast':
        res = autocontrast(img)

    elif args.command == 'fixinterlace':
        res = fixinterlace(img)

    # сохранить результат
    res = np.clip(res, 0, 1)  # обрезать всё, что выходит за диапазон [0, 1]
    res = (res * 255).astype(np.uint8)  # конвертация в байты
    skimage.io.imsave(args.output_file, res)
