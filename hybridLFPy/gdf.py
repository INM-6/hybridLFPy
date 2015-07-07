"""
Create db from gdf files to be able to then select neuron spike times.

Best use case:
1. run simulation
2. create sqlite db of spike times with indexing
3. use this db many times

Creating index for db will dominate insertions for larger set of spike times.

[TODO] Check how much slower things are if index is created at start
[TODO] Simplify block_read. sqlite probably can covert for insert.
[TODO] Error checking
[TODO] sqlite optimizations
[TODO] Create read buffers once instead of for each file
[TODO] Spike times may be storable as int32 rather than float, save space
"""
import numpy as np
import sqlite3 as sqlite
import os, glob
from time import time as now
import matplotlib.pyplot as plt


plt.rcdefaults()
plt.rcParams.update({
    'font.size' : 16,
    'axes.labelsize' : 16,
    'axes.titlesize' : 16,
    'legend.fontsize' : 14,
    'xtick.labelsize' : 16,
    'ytick.labelsize' : 16,
    'figure.subplot.wspace' : 0.3,
    'figure.subplot.hspace' : 0.3,
})

class GDF(object):
    """
    1. Read from gdf files.
    2. Create sqlite db of (neuron, spike time).
    3. Query spike times for neurons.


    Parameters
    ----------
    dbname : str
        Filename of sqlite database, see `sqlite3.connect`
    bsize : int
        Number of spike times to insert.
    new_db : bool
        New database with name dbname, will overwrite
        at a time, determines memory usage.
        
    
    Returns
    -------
    `hybridLFPy.gdf.GDF` object
    
    
    See also
    --------
    sqlite3, sqlite3.connect, sqlite3.connect.cursor
    """

    def __init__(self, dbname, bsize=int(1e6), new_db=True,
                 debug=False):
        """
        1. Read from gdf files.
        2. Create sqlite db of (neuron, spike time).
        3. Query spike times for neurons.
    
    
        Parameters
        ----------
        dbname : str
            Filename of sqlite database, see `sqlite3.connect`
        bsize : int
            Number of spike times to insert.
        new_db : bool
            New database with name dbname, will overwrite
            at a time, determines memory usage.
            
        
        Returns
        -------
        `hybridLFPy.gdf.GDF` object
        
        
        See also
        --------
        sqlite3, sqlite3.connect, sqlite3.connect.cursor
        """
        if new_db:
            try:
                os.unlink(dbname)
            except:
                print('creating new database file %s' % dbname)

        self.conn = sqlite.connect(dbname)
        self.cursor = self.conn.cursor()
        self.bsize = bsize
        self.debug = debug


    def _blockread(self, fname):
        """
        Generator yields bsize lines from gdf file.
        Hidden method.


        Parameters
        ----------
        fname : str
            Name of gdf-file.
            
        
        Yields
        ------
        list
            file contents
            
        """
        with open(fname, 'rb') as f:
            while True:
                a = []
                for i in range(self.bsize):
                    line = f.readline()
                    if not line: break
                    a.append(line.split())
                if a == []: raise StopIteration
                yield a

    def create(self, re='brunel-py-ex-*.gdf', index=True):
        """
        Create db from list of gdf file glob


        Parameters
        ----------
        re : str
            File glob to load.
        index : bool
            Create index on neurons for speed.
                    
        
        Returns
        -------
        None
        
        
        See also
        --------
        sqlite3.connect.cursor, sqlite3.connect
        
        """
        self.cursor.execute('CREATE TABLE IF NOT EXISTS spikes (neuron INT UNSIGNED, time REAL)')

        tic = now()
        for f in glob.glob(re):
            print(f)
            while True:
                try:
                    for data in self._blockread(f):
                        self.cursor.executemany('INSERT INTO spikes VALUES (?, ?)', data)
                        self.conn.commit()
                except:
                    continue
                break                

        toc = now()
        if self.debug: print('Inserts took %g seconds.' % (toc-tic))

        # Optionally, create index for speed
        if index:
            tic = now()
            self.cursor.execute('CREATE INDEX neuron_index on spikes (neuron)')
            toc = now()
            if self.debug: print('Indexed db in %g seconds.' % (toc-tic))


    def create_from_list(self, re=[], index=True):
        """
        Create db from list of arrays.


        Parameters
        ----------
        re : list
            Index of element is cell index, and element `i` an array of spike times in ms.
        index : bool
            Create index on neurons for speed.
        
        
        Returns
        -------
        None
        
        
        See also
        --------
        sqlite3.connect.cursor, sqlite3.connect
        
        """
        self.cursor.execute('CREATE TABLE IF NOT EXISTS spikes (neuron INT UNSIGNED, time REAL)')

        tic = now()
        i = 0
        for x in re:
            data = list(zip([i] * len(x), x))
            self.cursor.executemany('INSERT INTO spikes VALUES (?, ?)', data)
            i += 1
        self.conn.commit()
        toc = now()
        if self.debug: print('Inserts took %g seconds.' % (toc-tic))

        # Optionally, create index for speed
        if index:
            tic = now()
            self.cursor.execute('CREATE INDEX neuron_index on spikes (neuron)')
            toc = now()
            if self.debug: print('Indexed db in %g seconds.' % (toc-tic))


    def select(self, neurons):
        """
        Select spike trains.


        Parameters
        ----------
        neurons : numpy.ndarray or list
            Array of list of neurons.


        Returns
        -------
        list
            List of numpy.ndarray objects containing spike times.


        See also
        --------
        sqlite3.connect.cursor
        
        """
        s = []
        for neuron in neurons:
            self.cursor.execute('SELECT time FROM spikes where neuron = %d' % neuron)
            sel = self.cursor.fetchall()
            spikes = np.array(sel).flatten()
            s.append(spikes)
        return s


    def interval(self, T=[0, 1000]):
        """
        Get all spikes in a time interval T.


        Parameters
        ----------
        T : list
            Time interval.


        Returns
        -------
        s : list
            Nested list with spike times.



        See also
        --------
        sqlite3.connect.cursor
        
        """
        self.cursor.execute('SELECT * FROM spikes WHERE time BETWEEN %f AND %f' % tuple(T))
        sel = self.cursor.fetchall()
        return sel


    def select_neurons_interval(self, neurons, T=[0, 1000]):
        """
        Get all spikes from neurons in a time interval T.


        Parameters
        ----------
        neurons : list
            network neuron indices
        T : list
            Time interval.

        
        Returns
        ----------
        s : list
            Nested list with spike times.

        
        See also
        --------
        sqlite3.connect.cursor
        
        """
        s = []
        for neuron in neurons:
            self.cursor.execute('SELECT time FROM spikes WHERE time BETWEEN %f AND %f and neuron = %d'  % (T[0], T[1], neuron))
            sel = self.cursor.fetchall()
            spikes = np.array(sel).flatten()

            s.append(spikes)

        return s


    def neurons(self):
        """
        Return list of neuron indices.


        Parameters
        ----------
        None
        

        Returns
        -------
        list
            list of neuron indices
        
        
        See also
        --------
        sqlite3.connect.cursor
        
        """
        self.cursor.execute('SELECT DISTINCT neuron FROM spikes ORDER BY neuron')
        sel = self.cursor.fetchall()
        return np.array(sel).flatten()


    def num_spikes(self):
        """
        Return total number of spikes.


        Parameters
        ----------
        None
        
        
        Returns
        -------
        list

        """
        self.cursor.execute('SELECT Count(*) from spikes')
        rows = self.cursor.fetchall()[0]
        # Check against 'wc -l *ex*.gdf'
        if self.debug: print('DB has %d spikes' % rows)
        return rows


    def close(self):
        """
        Close `sqlite3.connect.cursor` and `sqlite3.connect` objects
        
        
        Parameters
        ----------
        None
        
        
        Returns
        -------
        None
        
        
        See also
        --------
        sqlite3.connect.cursor, sqlite3.connect

        """
        self.cursor.close()
        self.conn.close()


    def plotstuff(self, T=[0, 1000]):
        """
        Create a scatter plot of the contents of the database,
        with entries on the interval T.


        Parameters
        ----------
        T : list
            Time interval.
        
        
        Returns
        -------
        None
        
        
        See also
        --------
        GDF.select_neurons_interval
        """

        fig = plt.figure(figsize=(10,10))

        ax = fig.add_subplot(111)

        neurons = self.neurons()
        i = 0
        for x in self.select_neurons_interval(neurons, T):
            ax.plot(x, np.zeros(x.size) + neurons[i], 'o',
                    markersize=1, markerfacecolor='k', markeredgecolor='k',
                    alpha=0.25)
            i += 1
        ax.set_xlabel('time (ms)')
        ax.set_ylabel('neuron ID')
        ax.set_xlim(T[0], T[1])
        ax.set_ylim(neurons.min(), neurons.max())
        ax.set_title('database content on T = [%.0f, %.0f]' % (T[0], T[1]))


def test1():
    """ Need have a bunch of gdf files in current directory.
    Delete old db.
    """
    os.system('rm test.db')

    # Create db from excitatory files
    gdb = GDF('test.db', debug=True)
    gdb.create(re='brunel-py-ex-*.gdf', index=True)

    # Get spikes for neurons 1,2,3
    spikes = gdb.select([1,2,3])

    """ Wont get any spikes for these neurons
    cause they dont exist"""
    bad = gdb.select([100000,100001])

    gdb.close()

    print(spikes)
    print(bad)

if __name__ == '__main__':
    #test1()
    import doctest
    doctest.testmod()
