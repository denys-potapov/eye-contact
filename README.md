# eye-contact

## Prerequirments

You need python and cmake:
    
    sudo apt install python cmake

Load project deps:

    pip install -r requirements.txt

Download and uncompress pretrained face model:

    wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
    bzip2 -d shape_predictor_68_face_landmarks.dat.bz2

## Preview

    python3 preview.py open.jpg video.webm out.avi --vertical

## Live

Inslall v4l2loopback

    sudo apt install v4l2loopback-dkms

Enable and create eye opener camera:

    sudo modprobe v4l2loopback card_label="Eye contact" exclusive_caps=1

This shoud create new video device (`/dev/video1` usually).

Capture your open eyes photo from real camera:
    
    python3 capture.py open.jpg /dev/video0 

Runt it:

    python3 eye_contact.py open.jpg /dev/video0 /dev/video1

