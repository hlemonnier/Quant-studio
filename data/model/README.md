Model Generation
================

In order to create the full data model for backtest database, one need to use the script `generate.py`. Execution of this script will output a `model.sql` file that can be used in Postgres RDBM (WARNING: Need timescaledb extension installed and enabled for current database)

## Generate Schema File

First step consists in editing `generate.py` and set global variables *INSTUMENTS_LIST* and *SCHEMA_NAME* according to your own prefs. Then you just run: 

```bash
python3 generate.py
```

This execution will yield a `model.sql` file as output, that you'll be able to use in your TimescaleDB-enabled database (see below for TimescaleDB setup)

## Notes

* Schema is structured as such as there's one ohlcv table per instrument.
A timescaleDB hyper_table is then created from it, then data is aggregated on different time frames using 
TimescaleDB continously aggregated materialized views. 
* Schema also contains one table of trades per instrument (Possibly need to to have TSDB hyper_table too).

## TimescaleDB Extension setup + enabling in database

Following section breiefly explains how to install and configure TimescaleDB in multiple working environments.

### Windows:

Get Zip Archive from https://github.com/timescale/timescaledb/releases/download/2.18.2/timescaledb-postgresql-[[VERSION]]-windows-amd64.zip , extract it and run setup.exe

Restart Postgresql server.

*Note: Replace [[VERSION]] by the version of your own RDBM (15/16/17/..).*

### Ubuntu:

If your path to the `pg_config.exe` file is not a System PATH, the installation for TimescaleDB will fail, you will see an error:

```bash
 ERROR: could not get pg_config: exec: "pg_config": executable file not found in %PATH%
```

The path for the pg_config.exe file is usually:  `C:\Program Files\PostgreSQL\<version>\bin`

Once you set up the path to `pg_config.exe` as a system path the setup will then prompt you to enter the path to the `postgresql.conf` file.

The path for the `postgresql.conf` is usually located:  `C:\Program Files\PostgreSQL\<version>\data\postgresql.conf`  

As a default setup, say yes to every option.

If you get an error that looks like this:

```bash
ERROR: could not copy file 'timescaledb-2.18.2.dll': open C:/PROGRA~1/POSTGR~1/17/lib/timescaledb-2.18.2.dll: Access is denied.
```

Close the `setup.exe` and run it as an administrator, this should fix the problem.

Make sure to close all instances command prompt before starting the installation.

Make sure to check that the Postgresql service is running before trying to connecct to a local postgres server. Check services.msc on Windows and see if the instance of  
PostgreSQL is running. If not, start it 


### Enable TimescaleDB Extension in Postgresql:

Connect to database then launch following query:

```sql
CREATE EXTENSION IF NOT EXISTS timescaledb;
``` 

To confirm that extension is properly enabled, use:

```sql
SELECT * FROM pg_extension;
``` 

You should then see the row corresponding to timescaledb.

## Python Requirements (generator)

* jinja2
