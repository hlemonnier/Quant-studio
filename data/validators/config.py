'''
Script to read and load configuration settings from a specified INI file, particularly for connecting to a PostgreSQL database. 
'''


from configparser import ConfigParser

def load_config(filename="C:\\Users\\SolalDanan\\Coinbase API\\OHLCV\\database_ohlcv_btc_usd.ini", section='postgresql'):
    parser = ConfigParser()
    parser.read(filename)

    # get section, default to postgresql
    config = {}
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            config[param[0]] = param[1]
    else:
        raise Exception('Section {0} not found in the {1} file'.format(section, filename))

    return config

if __name__ == '__main__':
    config = load_config()
    print(config)


    