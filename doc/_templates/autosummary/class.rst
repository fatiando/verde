{{ fullname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

{% for item in methods %}
{% if item != '__init__' %}
.. automethod:: {{ objname }}.{{ item }}
{% endif %}
{% endfor %}

.. raw:: html

     <div style='clear:both'></div>

