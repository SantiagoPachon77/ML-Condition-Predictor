#################################################################################
# GLOBALS                                                                       #
#################################################################################

PYTHON_INTERPRETER = python
APP = main.py

#################################################################################
# COMMANDS                                                                      #
#################################################################################

health:
	@echo "Hello World!"

# target for creating a virtual environment- local development
setup_venv: requirements.txt
	@echo "Creating virtual environment with dependencies ..."
	@$(PYTHON_INTERPRETER) -m venv venv
	@venv/bin/pip install -r requirements.txt
	@echo "Done. To activate the virtual environment, run: source venv/bin/activate"

# remove virtual environment
clean_venv: venv
	@echo "Removing virtual environment ..."
	@rm -rf venv
	@echo "Done."

# -----------------------------------------
processed_data_products:
	@$(PYTHON_INTERPRETER) $(APP) processed-data-products

feaure_engineering_products:
	@$(PYTHON_INTERPRETER) $(APP) feaure-engineering-products

model_training:
	@$(PYTHON_INTERPRETER) $(APP) model-training

predict:
	@$(PYTHON_INTERPRETER) $(APP) predict
