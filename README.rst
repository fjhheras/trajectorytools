###############
trajectorytools
###############

⚠️ This it **not** the official repository of trajectorytools anymore. It migrated to `Polavieja's group GitLab <https://gitlab.com/polavieja_lab/trajectorytools/>`_ ⚠️

Installation
============

From PyPI
---------

.. code-block:: bash

    pip install trajectorytools


From Source
-----------

To clone this repository:

.. code-block:: bash

    git clone https://github.com/fjhheras/trajectorytools

To install requirements:

.. code-block:: bash

    pip install -r requirements.txt

To install the package:

.. code-block:: bash

    pip install .

or alternatively, locally with a symlink:

.. code-block:: bash

    pip install -e .

If you see this error: "gcc: fatal error: cannot execute ‘cc1plus’:
execvp: No such file or directory" you need the GNU C++ compiler.
To install it in, for example, Ubuntu and derivatives:

.. code-block:: bash

    sudo apt install g++


Example
==========

.. code-block:: python

    import numpy as np
    import matplotlib as mpl

    import trajectorytools as tt
    import trajectorytools.animation as ttanimation
    import trajectorytools.socialcontext as ttsocial
    from trajectorytools.constants import test_raw_trajectories_path

    # Loading test trajectories as a numpy array of locations
    test_trajectories = np.load(test_raw_trajectories_path, allow_pickle=True)

    # We will process the numpy array, interpolate nans and smooth it.
    # To do this, we will use the Trajectories API
    smooth_params = {'sigma': 1}
    traj = tt.Trajectories.from_positions(test_trajectories,
                                          smooth_params=smooth_params)

    # We assume a circular arena and populate center and radius keys
    center, radius = traj.estimate_center_and_radius_from_locations()

    # We center trajectories around the estimated center
    traj.origin_to(center)

    # We will normalise the location by the radius:
    traj.new_length_unit(radius)

    # We will change the time units to seconds. The video was recorded at 32
    # fps, so we do:
    traj.new_time_unit(32, 'second')

    # Now we can find the smoothed trajectories, velocities and accelerations
    # in traj.s, traj.v and traj.a
    # We can use, for instance, the positions in traj.s and find the border of
    # the group:
    in_border = ttsocial.in_alpha_border(traj.s, alpha=5)

    # Animation showing the fish on the border
    colornorm = mpl.colors.Normalize(vmin=0,
                                     vmax=3,
                                     clip=True)
    mapper = mpl.cm.ScalarMappable(norm=colornorm, cmap=mpl.cm.RdBu)
    color = mapper.to_rgba(in_border)

    anim1 = ttanimation.scatter_vectors(traj.s, velocities=traj.v, k=0.3)
    anim2 = ttanimation.scatter_ellipses_color(traj.s, traj.v, color)
    anim = anim1 + anim2

    anim.prepare()
    anim.show()


In the `directory examples`_, you can find some more example scripts.
Scripts use some example trajectories, which can be found in `data`_.
All example trajectories were obtained using idtracker.ai on videos
recorded in de Polavieja Lab (Champalimaud Research, Lisbon)

.. _directory examples: trajectorytools/examples
.. _data: trajectorytools/data

---
**NOTE**

Note that, when using constructors like `from_idtrackerai` and `from_positions`,
we need to calculate velocity and accelerations from positions. As a result, the
`traj` object has 2 frames less than the original positions array. By default, the
missing frames correspond to the first and last frames of the video. If you used
the option `"only_past":True` in `smooth_params`, the missing frames correspond
to the first two frames of the video.

---

Project maintainers
===================

Francisco J.H. Heras (2017-)
Francisco Romero Ferrero (2017-)
Dean Rance (2021-)

Contribute
==========

We welcome contributions. The preferred way to report problems is by creating an issue. The best way to propose changes in the code is to create a pull request. Please, check our `contribution guidelines`_ and our `code of conduct`_.

.. _contribution guidelines: .github/CONTRIBUTING.md
.. _code of conduct: .github/CODE_OF_CONDUCT.md


License
=======

This project is licensed under the terms of the GNU General Public License v3.0 (See COPYING). This means that you may copy, distribute and modify the software as long as you track changes/dates in source files. However, any modifications to GPL-licensed code must also be made available under the GPL along with build & install instructions.

If you use this work in an academic context and you want to acknowledge us, please cite some of the relevant papers:

Romero-Ferrero, F., Bergomi, M. G., Hinz, R. C., Heras, F. J., & de Polavieja, G. G. (2019). idtracker.ai: tracking all individuals in small or large collectives of unmarked animals. Nature methods, 1

Heras, F. J., Romero-Ferrero, F., Hinz, R. C., & de Polavieja, G. G. (2019). Deep attention networks reveal the rules of collective motion in zebrafish. PLoS computational biology, 15(9), e1007354.
