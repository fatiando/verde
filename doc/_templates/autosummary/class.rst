{{ fullname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

.. include:: backreferences/{{ fullname }}.examples

.. raw:: html

     <div style='clear:both'></div>

{% for item in methods %}
{% if item != '__init__' %}
.. automethod:: {{ objname }}.{{ item }}
{% endif %}
{% endfor %}

.. raw:: html

     <div style='clear:both'></div>

