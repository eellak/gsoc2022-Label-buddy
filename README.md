# Google Summer of Code with GFOSS :sun_with_face: 

**Project:** Automateda Audio-Tagging Using Transfer Learning

**Mentors:** Pantelis Vikatos, Agisilaos Kounelis, Ioannis Sina

**Past Mentors:** Markos Gogoulos

**Contributor:** Ioannis Prokopiou

**Past Contributor:** Ioannis Sina

# Introduction

An annotation tool helps people (without the need for specific knowledge) to mark a segment of an audio file (waveform), an image or text etc. in order to specify the segment’s properties. Annotation tools are used in machine learning applications such as Natural Language Processing (NLP) and Object Detection in order to train machines to identify objects or text. While there is a variety of annotation tools, most of them lack the multi-user feature (multiple users annotating a single project simultaneously) whose implementation is planned in this project. The audio annotation process is usually tedious and time consuming therefore, these tools (annotation tools which provide the multi-user feature) are necessary in order to reduce the effort needed as well as to enhance the quality of annotations. Since in most tasks related to audio classification, speech recognition, music detection etc., machine and deep learning models are trained and evaluated with the use audio that has previously been annotated by humans, the implementation of such a tool will lead to higher accuracy of annotated files, as they will have been annotated by more than one human, providing a more reliable dataset. In effect, multi-user annotation will reduce the possibility of human error e.g. an occasional mistaken labelling of a segment might be pointed out by another annotator.

AI-assisted solutions have emerged to speed up annotation workflows and increase productivity. Deep learning models can be used for annotation and can kickstart your development effort by enabling faster annotation of datasets for AI algorithms. Deep learning models are sensitive to the data used to train them, this makes it hard to train the deep learning models on a specific dataset and deploy them on a different dataset. As a solution, transfer learning for sound could help adapt pretrained models into various datasets. Deep learning models used for annotation can be tuned and improved by retraining these pretrained models based on new datasets.

**Already existing annotation tools:**

Label Studio: https://github.com/heartexlabs/label-studio

BAT annotation tool: https://github.com/BlaiMelendezCatalan/BAT

Computer Vision Annotation Tool (CVAT): https://github.com/openvinotoolkit/cvat

# Project goals :dart: 

This project is an enhancement to the previous work that has been done previously. Its goal is to make annotation simple and easy while also
maintaining a well-defined manager-annotator-reviewer framework. The goal of this project is to use Transfer Learning (TL) approaches to make the
annotation process easier for the user by offering label predictions. This way it will be possible to accomplish more with less data and effort. It is manly devided in two categories of tasks: Machine/Transfer Learning and Django.

**Machine/Transfer Learning:**

* Conduct research for the appropriate model architecture
* Modify the annotation process by integrating the model
* Test the model by providing evaluation metrics

**Django:**

* Add lazy loading for the audio files: load segments of the file when needed (i.e., YouTube). This will lead to better performance when the audio file is too big.
* Add Django Testing
* Dockerization
* Add documentation
* Add rar file upload functionality - currently, users can only upload zip files (optional)
* UI improvements (optional)

# Steps to run

Clone repository and cd to the folder
~~~
git clone https://github.com/eellak/gsoc2021-audio-annotation-tool/
cd gsoc2021-audio-annotation-tool
~~~

Create virtual enviroment
~~~
python3 -m venv env
~~~

Activate it for **Linux**
~~~
source env/bin/activate
~~~

Activate it for **Windows**
~~~
env\Scripts\activate
~~~

Install audiowaveform
~~~
Follow the steps described [here](https://github.com/bbc/audiowaveform#installation).
~~~

Install requirements and cd to label_buddy/
~~~
pip install -r requirements.txt
cd label_buddy
~~~

Make migrations for the Database and all dependencies
~~~
python manage.py makemigrations users projects tasks
python manage.py migrate
~~~

After the above process create a super user and run server
~~~
python manage.py createsuperuser
python manage.py runserver
~~~

Visit http://localhost:8000/admin, navigate to users/[your user] and set can_create_projects to true so you can start creating projects.

Visit https://labelbuddy.io/ and sign with the following credentials:

  - **Username**: demo
  - **Password**: labelbuddy123

in order to create projects, upload files and annotate them.
