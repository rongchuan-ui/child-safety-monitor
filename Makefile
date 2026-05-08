PYTHON ?= python

.PHONY: install run run-mobile run-object run-zone check clean

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

run:
	$(PYTHON) combined_monitor.py

run-mobile:
	$(PYTHON) mobile_camera_monitor.py

run-object:
	cd object_contact_alert/app && $(PYTHON) object_contact_web.py

run-zone:
	cd restricted_zone_alert && $(PYTHON) zone_alert_web.py

check:
	$(PYTHON) -m py_compile combined_monitor.py mobile_camera_monitor.py object_contact_alert/app/object_contact_core.py object_contact_alert/app/object_contact_web.py restricted_zone_alert/zone_alert_core.py restricted_zone_alert/zone_alert_web.py

clean:
	rm -rf __pycache__ object_contact_alert/app/__pycache__ restricted_zone_alert/__pycache__ .certs
