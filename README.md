# REPARTO DE TAREAS
## **BILBO**

### In progress:
    - Measurement of the probability of a point's being: Boltzmann's statistics + Hooke's law.
        - Theorical approach already completed
        - How to calculate the mean energy of the system? https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.ewm.html

### Done:
    - Tendency checker based on first and second derivatives.

---
## **MOLI**

### To do:
- Production extraction implementation:
    
    1. Define the data model. [ - ]
    2. Write CRUD senteces [ - ]
    3. Implement full extraction [ - ]
    4. Populate database with old data [ - ]
    5. Implement incremental extraction [ - ]


- Read about Alpaca paper trading.
### In progress:
- Check how to calculate Sharpe Ratio with VectorBT

### Done:
- Crear una función para extraer datos a nivel de 5 minutos utilizando la API de interactive brokers con un buen historico (alrededor de 6 años estaría bien) [ &#10004; ]
- Una vez creada la función extraer datos de diferentes activos y guardarlos en una carpeta en formato csv. (top empresas s&p500, oro y plata, bitcoin y ethereum, divisas como USD-EUR. [  &#10004; ]

### Blocked

- Understand and implement IB extraction with its software (Cannot be done until the account is set up).

---
## **UNAI**

### To do:
- Test method using EWM for denoising
- Find ways to avoid or reduce overfitting
- Check if we can apply taxes to the gains of the bot for simulation
- Investigate how we can optimize a portfolio based on our strategy
- Create benchmark to compare our strategy with different strategies (random, buy and hold, etc)

### In progress:
- Prepare vectorbt pipeline to allow multiple asset optimization and plotting

### Done:
- Modify plot_optimization and step_validation so it can work with folds
- Bayesian optimization using optuna
- Implement derivatives strategy
- Implement pipeline with vectorbt
