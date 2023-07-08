#Urban Gardens

## 1 Introduction
Community shared urban gardens are collaborative
spaces where individuals from a community can come
together to cultivate and maintain gardens within urban areas. These gardens provide a shared space for
growing fruits, vegetables, herbs, and flowers. Urban
gardening promotes sustainability as they allow communities to produce their own food locally, reducing
the need to transport food for long distances. It improves health as gardening is proved to be a good
way of exercising and improving mental health, also
they give access to healthy fresh produce to consume
promoting a healthier diet. Community shared urban
growing gardens provide a space for community members to come together and work towards a common
goal. In an urbanized world, community gardens connect urban residents and constitute spaces for a commons management of urban resources. Urban gardening also have many environmental benefits, it decreases
some pollution caused by food transportation, it combats the urban heat island effect by absorbing sunlight
and providing shades, and decreases water runoff by
absorbing rainfall. These gardens can also be an opportunity to educate people on food production and
consumption, while transforming unused urban spaces
into beautiful green areas. There are however some
challenges that have to be considered like water resource availability or sunlight and weather conditions.
But the main one is the limited amount of land available for this use inside of a city. This last challenge is
the one this project focuses on. The aim of this project
was to build an Artificial Intelligence tool capable of
choosing the best plots of land to be used for urban gardening whilst minimising the necessary amount of land
area used. It also must satisfy a minimal produce goal
and minimize the distance of the plots to the assigned
community. To do so it will take into account the city
size, population and geographical location. The idea is
that this tool could be used by a city government to
help plan and optimize the construction of many urban gardens inside of a city. In addition it create a
list of plants which are best considering the geographic
location.


## 2 Model

To solve this problem a model to optimize is needed.
The idea is to do so as a constraint satisfaction model.
Firstly we assume that each city has a certain number of plots that are available to be chosen for this
purpose, each one of these in a different location and
of different sizes. For this project these are randomly
generated, however the idea is that in a real application a city government draws up a list of plots with
location and size. This is also the case for the neighborhoods, in which case the number of neighborhoods
generated depends on the city land size and the number of people in each neighborhoods depends on the
city population. Taking for example the city of OSLO
45 different neighborhoods and 122 possible plots were
generated.

![Generated plots and neighborhoods for the
city of OSLO](imgs\plotOslo.png)

