install latest python

install lib with command below
    pip install torch torchvision torchaudio
    pip install matplotlib
    pip install IPython
    pip install numpy==1.26.4

note:
    user lower version of numpy, higher version produce warning and error .
    torch will use cpu to run the training.
    to use gpu, install cuda and install pytorch with cuda support (windows only).

model_25 is trained with 25 epochs (better performance).