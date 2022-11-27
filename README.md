# Intelligent-Systems-Development-

MTS course at the university ITMO

Документация от FastApi на своем серваке
`http://127.0.0.1:8000/docs`

Тестовые прогоны на севаке по уроку

```python
http://127.0.0.1:8000/reco/kek/12
{"user_id":12,"items":[0,1,2,3,4,5,6,7,8,9]}
```

На убунту20.04 rectools встала после (конкретно проблему с implicit=0.4.4 фиксит):

```python
sudo apt-get install g++ 
poetry add rectools
```

Для теста нужно просто запустить в консоли pytest, пока он только проверить /health
