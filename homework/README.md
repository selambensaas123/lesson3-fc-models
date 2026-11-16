# Урок 3. Полносвязные сети

Условия работы приведены в [HOMEWORK.md](../HOMEWORK.md). Эксперименты используют конфигурируемую модель из примеров преподавателя.

Из корня проекта:

```powershell
python -m pip install torch torchvision matplotlib
python -m unittest -v
python -m homework.homework_depth_experiments
python -m homework.homework_width_experiments
python -m homework.homework_regularization_experiments
python -m homework.build_report
```

Для набора данных используются 12 000 обучающих и 2 000 валидационных изображений из официального обучающего набора с фиксированным seed 42. Официальная тестовая выборка используется для отчётных кривых. Сетка ширины оценивается по валидации. Для ширины три заданных числа соответствуют скрытым слоям, после которых расположен классификатор.

Графики и метрики сохраняются в homework/plots и homework/results.
