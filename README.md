# Is There Water?
### Tanzania
<div>
<P ALIGN=CENTER><img src="images/watertap.jpg" style="display: block; margin-left: auto; margin-right: auto;"  width="900"/></P></div>

* A social impact driven tech firm hosts open competitions to crowdsource data science solutions to social challenges, with potential utility by organizations and institutions taking on those challenges. The online challenges last a few months, the global community of data scientists can compete to provide the best statistical model for difficult predictive social impact problems. One of the competitions asks participants to predict whether or not any given water point - from a dataset with a variety of descriptive feature data on tens of thousands of water points - is functioning, in need of repair, or broken.
* Tanzania has achieved has averaged 6.5% economic growth over the past decade and is on its way to becoming a middle income country. But inequalities are entrenched, and the country has seen only a modest reduction in poverty over the same period. Some 40% of the country's population are able to rely on regular access to safe drinking water sources.
* In rural and underprivileged areas, improved water points are funded and installed by a wide array of actors - including the local government, civil society, international donors, private companies, and individuals - using a variety of technologies and water sourcing methods. Understanding when or how a water point might break or need maintenance could inform budget allocations and maintenance scheduling, ultimately optimizing access to safe water with those resources.

### The analysis questions:
* What are the key predictors of whether or not a water point is functioning on any given day?
* Is there a machine learning model that can identify with reasonable certainty which water points are likely to need maintenance or replacement?

### The dataset:
* The raw CSV dataset, provided by the online competition, has 59,400 rows (each representing a water point in Tanzania) and 40 columns of descriptive features, which include information on various levels of geographical location, population using the water point, water point management and payment schemes, type and age of the hardware, water pressure at the tap, and type of water source.
* The target (to be predicted) is a categorical column with three classes: 'functional,' 'functional needs repair,' and 'non functional.'
* The following represents the first 5,000 rows of the dataset, with the white lnies indicating missing data:
<p>
<img align="center" src="images/raw_msno_mtx.png" width="800">
</p>

## **Exploratory data analysis (EDA) & Feature Engineering**

[**Pandas dataframe profile**](http://htmlpreview.github.io/?https://github.com/mstyslinger/is_there_water_tz/blob/master/pandas_profile_reports/pfr_cleaned.html) 

