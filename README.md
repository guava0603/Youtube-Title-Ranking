# YouTube Video Title Scoring

This project aims to predict a score in 0~100 as the beneficial result of the title,
the higher score represents the higher probability to get higher views.
Moreover, the system will show some recommended titles by catching keywords in the given title.

## Technology Specification

- Pretrained model
- TF-IDF Analysis

## Prerequisite

1. Python 3.6^
2. Python flask environment

## How to ?

1. Run `python3 api.py`, and localhost:5000 will be listening
2. call `GET http://localhost:5000/search/<TITLE>`, you may get the score.
3. call `GET http://localhost:5000/article/<TITLE>`, you may get the recommended titles.

## Contributor

Guava Chen,
Yuki Yen,
Pei-Ying Li,
Yu-Ching Chen,
Hong-Ming Lai,
Min-Chi Chiang
