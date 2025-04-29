{{ fullname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

{% if attributes %}

Attributes
----------

{% for item in attributes %}
.. autoattribute:: {{ objname }}.{{ item }}
{% endfor %}

{% endif %}


{% if methods %}

Methods
-------

{% for item in methods %}
{% if item != '__init__' %}
.. automethod:: {{ objname }}.{{ item }}
{% endif %}
{% endfor %}

{% endif %}

.. raw:: html

     <div style='clear:both'></div>

