INSTUMENTS_LIST=['btc_usd_spot','eth_usd_spot','sol_usd_spot','ltc_usd_spot','btc_eur_spot','eth_eur_spot','sol_eur_spot','ltc_eur_spot']

SCHEMA_NAME="qresearch"

import jinja2

model = open('model.sql','w')
model_tpl =  open('model-template.sql','r').read()
tpl = jinja2.Template(model_tpl)

for inst in INSTUMENTS_LIST:
    rndr = tpl.render(schema=SCHEMA_NAME,instrument=inst)
    model.write(rndr)

model.close()



