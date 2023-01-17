import os

import pytest

# import sys
# sys.stdout, sys.stderr = os.devnull, os.devnull # silence command-line output temporarily
from pysces import model as PyscesModel
from pysces.PyscesInterfaces import Core2interfaces as PyscesCore2Interface

# sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__# unsilence command-line output
# Use __SILENT_START__ ?

SBML_FILE = "epic/example_models/sbml/sbml_file.xml"


# This only takes the path, not the file itself and creates a tempdir at the path stored in the variable "datafiles"
@pytest.mark.datafiles(SBML_FILE)
def test_sbml_file_exists(datafiles):
    """Test wether the given datafiles path contains the sbml file needed for the sbml model

    :param datafiles: _description_
    :type datafiles: _type_
    """
    path = str(datafiles)

    sbml_file = os.path.join(path, "sbml_file.xml")
    assert os.path.isfile(sbml_file)


@pytest.mark.datafiles(SBML_FILE)
def test_model_creation(datafiles):
    """Test wether pysces can convert the sbml to a psc file

    :param datafiles: The directory containing the sbml file
    :type datafiles: _type_
    """
    path = str(datafiles)

    interface = PyscesCore2Interface()
    interface.convertSBML2PSC(
        sbmlfile="sbml_file.xml",
        sbmldir=path,
        pscdir=path,
        pscfile="sbml_file.psc",
    )
    model = PyscesModel(File="sbml_file.psc", dir=path)
    assert model is not None


@pytest.mark.datafiles(SBML_FILE)
def test_sbml_model(datafiles):
    """Tests wether pysces can show the ODE of the model contained in the sbml file"""
    path = str(datafiles)
    interface = PyscesCore2Interface()
    interface.convertSBML2PSC(
        sbmlfile="sbml_file.xml",
        sbmldir=path,
        pscdir=path,
        pscfile="sbml_file.psc",
    )
    model = PyscesModel(File="sbml_file.psc", dir=path)

    print(model.showODE())
    assert True
