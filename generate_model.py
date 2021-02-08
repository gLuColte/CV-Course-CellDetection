import os
import shutil

from src.dataset import build_dataset
from src.train import train, test
from src.predict import predict
from src.preprocess import preprocess_dic, preprocess_phc, preprocess_fluo


def cleanup():
    shutil.rmtree("data", ignore_errors=True)
    shutil.rmtree("models", ignore_errors=True)
    shutil.rmtree("output", ignore_errors=True)

    os.mkdir("data")
    os.mkdir("data/bnd")
    os.mkdir("data/dst")
    os.mkdir("data/img")
    os.mkdir("data/seg")
    os.mkdir("models")
    os.mkdir("output")
    os.mkdir("output/test")


def train_dic(iter=10, scale=1.0):
    cleanup()
    shutil.rmtree("models_dic", ignore_errors=True)
    shutil.rmtree("output_dic", ignore_errors=True)

    build_dataset(
        base="COMP9517 20T2 Group Project Image Sequences/DIC-C2DH-HeLa", 
        kernel_sz=7, 
        scale=scale)

    os.remove("data/bnd/16.png")
    os.remove("data/dst/16.png")
    os.remove("data/img/16.png")
    os.remove("data/seg/16.png")

    train(max_iter=iter, prep_func=preprocess_dic)
    test(prep_func=preprocess_dic)

    os.rename("models", "models_dic")
    os.rename("output", "output_dic")
    shutil.move("log.txt", "models_dic/log.txt")


def train_phc(iter=10, scale=2.0):
    cleanup()
    shutil.rmtree("models_phc", ignore_errors=True)
    shutil.rmtree("output_phc", ignore_errors=True)

    build_dataset(
        base="RawData/PhC-C2DL-PSC",
        kernel_sz=3, 
        scale=scale)

    train(max_iter=iter, prep_func=preprocess_phc)
    test(prep_func=preprocess_phc, cpu=False)

    os.rename("models", "models_phc")
    os.rename("output", "output_phc")
    shutil.move("log.txt", "models_phc/log.txt")


def train_fluo(iter=10, scale=1.0):
    cleanup()
    shutil.rmtree("models_fluo", ignore_errors=True)
    shutil.rmtree("output_fluo", ignore_errors=True)

    build_dataset(
        base="RawData/Fluo-N2DL-HeLa",
        kernel_sz=3, 
        scale=scale)

    for i in range(36):
        if (i not in [0, 3, 16]):
            os.remove(f"data/bnd/{i}.png")
            os.remove(f"data/dst/{i}.png")
            os.remove(f"data/img/{i}.png")
            os.remove(f"data/seg/{i}.png")
        
    train(max_iter=iter, prep_func=preprocess_fluo, save_interval=100)
    test(prep_func=preprocess_fluo, cpu=False)

    os.rename("models", "models_fluo")
    os.rename("output", "output_fluo")
    shutil.move("log.txt", "models_fluo/log.txt")


def predict_dic(scale=1.0, cpu=True):
    os.makedirs("output_dic", exist_ok=True)
    os.makedirs("output_dic/Sequence 1", exist_ok=True)
    os.makedirs("output_dic/Sequence 2", exist_ok=True)
    os.makedirs("output_dic/Sequence 3", exist_ok=True)
    os.makedirs("output_dic/Sequence 4", exist_ok=True)

    predict(model_path="models_dic/",
            input_path="RawData/"
                        "DIC-C2DH-HeLa/Sequence 1",
            out_path="output_dic/Sequence 1",
            prep_func=preprocess_dic,
            scale=scale,
            cpu=cpu)

    predict(model_path="models_dic/",
            input_path="RawData/"
                        "DIC-C2DH-HeLa/Sequence 2",
            out_path="output_dic/Sequence 2",
            prep_func=preprocess_dic,
            scale=scale,
            cpu=cpu)

    predict(model_path="models_dic/",
            input_path="RawData/"
                        "DIC-C2DH-HeLa/Sequence 3",
            out_path="output_dic/Sequence 3",
            prep_func=preprocess_dic,
            scale=scale,
            cpu=cpu)

    predict(model_path="models_dic/",
            input_path="RawData/"
                        "DIC-C2DH-HeLa/Sequence 4",
            out_path="output_dic/Sequence 4",
            prep_func=preprocess_dic,
            scale=scale,
            cpu=cpu)


def predict_phc(scale=2.0, cpu=True):
    os.makedirs("output_phc", exist_ok=True)
    os.makedirs("output_phc/Sequence 1", exist_ok=True)
    os.makedirs("output_phc/Sequence 2", exist_ok=True)
    os.makedirs("output_phc/Sequence 3", exist_ok=True)
    os.makedirs("output_phc/Sequence 4", exist_ok=True)

    predict(model_path="models_phc/",
            input_path="RawData/"
                        "PhC-C2DL-PSC/Sequence 1",
            out_path="output_phc/Sequence 1",
            prep_func=preprocess_phc,
            scale=scale,
            cpu=cpu)

    predict(model_path="models_phc/",
            input_path="RawData/"
                        "PhC-C2DL-PSC/Sequence 2",
            out_path="output_phc/Sequence 2",
            prep_func=preprocess_phc,
            scale=scale,
            cpu=cpu)

    predict(model_path="models_phc/",
            input_path="RawData/"
                        "PhC-C2DL-PSC/Sequence 3",
            out_path="output_phc/Sequence 3",
            prep_func=preprocess_phc,
            scale=scale,
            cpu=cpu)

    predict(model_path="models_phc/",
            input_path="RawData/"
                        "PhC-C2DL-PSC/Sequence 4",
            out_path="output_phc/Sequence 4",
            prep_func=preprocess_phc,
            scale=scale,
            cpu=cpu)


def predict_fluo(scale=1.0, cpu=True):
    os.makedirs("output_fluo", exist_ok=True)
    os.makedirs("output_fluo/Sequence 1", exist_ok=True)
    os.makedirs("output_fluo/Sequence 2", exist_ok=True)
    os.makedirs("output_fluo/Sequence 3", exist_ok=True)
    os.makedirs("output_fluo/Sequence 4", exist_ok=True)

    predict(model_path="models_fluo/",
            input_path="RawData/"
                        "Fluo-N2DL-HeLa/Sequence 1",
            out_path="output_fluo/Sequence 1",
            prep_func=preprocess_fluo,
            scale=scale,
            cpu=cpu)

    predict(model_path="models_fluo/",
            input_path="RawData/"
                        "Fluo-N2DL-HeLa/Sequence 2",
            out_path="output_fluo/Sequence 2",
            prep_func=preprocess_fluo,
            scale=scale,
            cpu=cpu)

    predict(model_path="models_fluo/",
            input_path="RawData/"
                        "Fluo-N2DL-HeLa/Sequence 3",
            out_path="output_fluo/Sequence 3",
            prep_func=preprocess_fluo,
            scale=scale,
            cpu=cpu)

    predict(model_path="models_fluo/",
            input_path="RawData/"
                        "Fluo-N2DL-HeLa/Sequence 4",
            out_path="output_fluo/Sequence 4",
            prep_func=preprocess_fluo,
            scale=scale,
            cpu=cpu)


if __name__ == "__main__":
    # print("Training DIC model")
    # train_dic(iter=1500, scale=1.0)

    print("Building DIC predictions")
    predict_dic(scale=1.0, cpu=False)

    # print("Training PhC model")
    # train_phc(iter=750, scale=1.5)

    print("Building PhC predictions")
    predict_phc(scale=1.5, cpu=False)

    # print("Training Fluo model")
    # train_fluo(iter=100, scale=1.0)

    print("Building Fluo predictions")
    predict_fluo(scale=1.0, cpu=False)
