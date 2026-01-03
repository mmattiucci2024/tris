#############################################################################################
The followng instructions can be inserted in a txt file a directly loaded by a generci LLM.


Da ora in avanti la "matrice del Tris" è una tabella 3x3 che può contenere, in ogni posizione, solo i simboli X,O,_

Da ora in avanti la "linearizzazione orizzontale Tris" è un vettore unidimensionale contenente 3 vettori pari alle tre righe della "matrice del Tris".

Da ora in avanti la "linearizzazione verticale Tris" è un vettore unidimensionale contenente 3 vettori pari alle tre colonne della "matrice del Tris".

Da ora in avanti la "linearizzazione diagonale Tris" è un vettore unidimensionale contenente 2 vettori pari alle due diagonali della "matrice del Tris", prima la principale e poi quella secondaria.

Da ora in avanti un "Tris di X" in una "matrice del Tris" è quando la "linearizzazione orizzontale Tris" oppure la "linearizzazione verticale Tris" oppure la  "linearizzazione diagonale Tris" ha almeno un vettore al suo interno con tre X.

Da ora in avanti un "Tris di O" in una "matrice del Tris" è quando la "linearizzazione orizzontale Tris" oppure la "linearizzazione verticale Tris" oppure la  "linearizzazione diagonale Tris" ha almeno un vettore al suo interno con tre O.

Da ora in avanti un "Tris pieno" in una "matrice del Tris" è quando la "linearizzazione orizzontale Tris" e la "linearizzazione verticale Tris" e la  "linearizzazione diagonale Tris" hanno solo O e X.

Da ora in avanti un "Due X nel Tris" in una "matrice del Tris" è quando la "linearizzazione orizzontale Tris" oppure la "linearizzazione verticale Tris" oppure la  "linearizzazione diagonale Tris" ha almeno un vettore al suo interno con due X e un solo _.

Da ora in avanti un "Due O nel Tris" in una "matrice del Tris" è quando la "linearizzazione orizzontale Tris" oppure la "linearizzazione verticale Tris" oppure la  "linearizzazione diagonale Tris" ha almeno un vettore al suo interno con due O e un solo _.

Da ora in avanti il "cuore del Tris" è l'elemento centrale nel vettore centrale della "linearizzazione orizzontale Tris".

Conosci il gioco del tris per cui usa le sue regole ma aggiungi le seguenti strategie nello specifico ordine che segue:

(1) se c'è un "Tris di X" nella "matrice del Tris" ferma il gioco e asserisci di aver vinto.

(2) se c'è un "Tris di O" nella "matrice del Tris" ferma il gioco e asserisci che io ho vinto.

(3) se c'è un "Tris pieno" nella "matrice del Tris" ferma il gioco e asserisci che è patta.

(4) se c'è un "Due X nel Tris" o se c'è un "Due O nel Tris" nella "matrice del Tris" metti una X al posto della _ e poi mostrami la "matrice del Tris". 

(5) se il "cuore del Tris" contiene una _ allora mettici una X al posto della _ e poi mostrami la "matrice del Tris". 

(6) se il primo elemento del primo vettore della "linearizzazione orizzontale Tris" Ã¨ O e l'ultimo elemento dell'ultimo vettore della "linearizzazione orizzontale Tris" Ã¨ _ metti una X al posto della _ e poi mostrami la "matrice del Tris". 

(7) se l'ultimo elemento del primo vettore della "linearizzazione orizzontale Tris" è O e il primo elemento dell'ultimo vettore della "linearizzazione orizzontale Tris" è _ metti una X al posto della _ e poi mostrami la "matrice del Tris".

(8) se il primo elemento del terzo vettore della "linearizzazione orizzontale Tris" è O e l'ultimo elemento del primo vettore della "linearizzazione orizzontale Tris" è _ metti una X al posto della _ e poi mostrami la "matrice del Tris".

(9) se l'ultimo elemento del terzo vettore della "linearizzazione orizzontale Tris" è O e il primo elemento del primo vettore della "linearizzazione orizzontale Tris" è _ metti una X al posto della _ e poi mostrami la "matrice del Tris".

(10) se nel primo vettore della "linearizzazione orizzontale Tris" ci sono due X e nel secondo vettore della "linearizzazione orizzontale Tris" c'è almeno una _ nella stessa posizione di una delle X, metti una X al posto della _ e poi mostrami la "matrice del Tris".

(11) se nel primo vettore della "linearizzazione orizzontale Tris" ci sono due X e nel terzo vettore della "linearizzazione orizzontale Tris" c'è almeno una _ nella stessa posizione di una delle X, metti una X al posto della _ e poi mostrami la "matrice del Tris".

(12) se nel secondo vettore della "linearizzazione orizzontale Tris" ci sono due X e nel primo vettore della "linearizzazione orizzontale Tris" c'è almeno una _ nella stessa posizione di una delle X, metti una X al posto della _ e poi mostrami la "matrice del Tris".

(13) se nel secondo vettore della "linearizzazione orizzontale Tris" ci sono due X e nel terzo vettore della "linearizzazione orizzontale Tris" c'è almeno una _ nella stessa posizione di una delle X, metti una X al posto della _ e poi mostrami la "matrice del Tris".

(14) se nel terzo vettore della "linearizzazione orizzontale Tris" ci sono due X e nel primo vettore della "linearizzazione orizzontale Tris" c'è almeno una _ nella stessa posizione di una delle X, metti una X al posto della _ e poi mostrami la "matrice del Tris".

(15) se nel terzo vettore della "linearizzazione orizzontale Tris" ci sono due X e nel secondo vettore della "linearizzazione orizzontale Tris" c'è almeno una _ nella stessa posizione di una delle X, metti una X al posto della _ e poi mostrami la "matrice del Tris".

(16) se in uno dei vettori della "linearizzazione orizzontale Tris" c'è una X e due _, metti una X al posto di una delle _ che sono all'inizio o alla fine del vettore e poi mostrami la "matrice del Tris".

(17) se in uno dei vettori della "linearizzazione verticale Tris" c'è una X e due _, metti una X al posto di una delle _ che sono all'inizio o alla fine del vettore e poi mostrami la "matrice del Tris".

(18) se in uno dei vettori della "linearizzazione diagonale Tris" c'è una X e due _, metti una X al posto di una delle _ che sono all'inizio o alla fine del vettore e poi mostrami la "matrice del Tris".

(19) se arrivi a questa regola scegli una posizione a caso nella "matrice del Tris" dove ci sia una _, metti una X al posto di una delle _ e poi mostrami la "matrice del Tris".
