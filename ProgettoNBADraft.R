library(tidyr)
library(dplyr)
library(readr)
library(RColorBrewer)
library(tidyquant)
library(tidyverse)
library(tidyqu)
library(ggplot2)

# THE NBA DATASET
setwd("~/Google Drive/Data Science/I anno/II semestre/STATISTICAL LEARNING 2 (MOD. B)/ProgettoNBA")
#df <- read.csv("dataset_all_seasons.csv")#%>% filter(start_season>2009)
#df <- read.csv("dataset_final_20102020.csv")
df <- read.csv("dataset_final.csv")


# We add a column named 'num_seasons' which count the number of season a player
# has played already
#df <- df %>% mutate(df, num_seasons = nchar(seasons)%/%6)

# We select the features we want to do prediction on 
# This includes the 'drafted' column
(feat_names = colnames(df))

feat_prediction = c("height", "weight", "min_per", "field_goal", 
                    "field_attmps","field_pct", "two_pointer", "two_pattamps", 
                    "two_pct", "three_ptrs", "three_pattmp", "three_pct", 
                    "free_throws", "free_attmps", "free_pct", "assists", 
                    "steals", "blocks", "points", "off_reb",  "def_reb", 
                    "total_reb", "turnovers", "drafted", "seasons")

# Python to R Boolean conversion
df$drafted <- as.logical(df$drafted)

# We select the meaningful columns only (using package dplyr)
df_fit <- df %>% select(all_of(feat_prediction))
# or we can filter for a role
df_fit <- df %>% filter(position == 'Center', start_season > 2010) %>% select(feat_prediction)
df_fit <- df %>% filter(position == 'Guard') %>% select(all_of(feat_prediction))
#df_fit <- df %>% filter(position == 'Forward') %>% select(all_of(feat_prediction))



df %>% group_by(position) %>% count()
table(df$position)


X <- as.matrix(select(df_fit,-'drafted'))
y <- 1*(df_fit$drafted)

sum(y)


# Applicazione di modello logistico
# We are not supposed to normalise/standardize data for applying a 
# logistic regression model. This is not the case for ridge/lasso
logit.out <- glm(y~X, family = binomial)
summary(logit.out)

plot(df_fit$assist, y)
plot(df_fit$num_seasons, y)
plot(df_fit$assist, y)


# We would like to see what are the proportions of players roles in a draft
# session. We could also see if they somehow correspond to those of the entire
# dataset

# Let's focus first on drafted players
df_roles = df %>% filter(drafted == TRUE)
n <- dim(df_roles)[1]
(df_roles %>% count(position))$n /n

# The proportion are essentially the same among drafted and undrafted players
df_roles = df 
n <- dim(df_roles)[1]
(df_roles %>% count(position))$n /n


# It would be interesting to see if these proportion change throughout the years
df_roles = df %>% group_by(end_season)
n <- dim(df_roles)[1]
df_boh <- df_roles %>% count(position) %>% 
  pivot_wider(names_from = "position", values_from = "n")
df_boh[,-1] <- df_boh[,-1] / (df_roles %>% count())$n

plot(df_boh$end_season, df_boh$Center)
plot(df_boh$end_season, df_boh$Forward, lty = 2)


prova <- ggplot(df_boh, aes(x=end_season, y = Center)) +
                scale_x_continuous(breaks=seq(2001, 2020, 1)) +
                geom_line(aes(y = Center, colour = "var3"))+ 
                geom_line(aes(y = Guard, colour = "var1"))+
                geom_line(aes(y = Forward, colour = "var2"))+
                scale_y_continuous(breaks=seq(0, 1, 0.1)) + ylim(c(0,1))+
                geom_line() + 
                #geom_ma(ma_fun = SMA, n = 4,col="firebrick") +
                #geom_smooth(method="lm") +
                labs(x = "Season", y = "Proportion in the dataset", 
                     title = "Proportion of players positions", 
                     subtitle = "Seasons from 2001 to 2020") 
prova


#################################################################
#                       Stacked Bar Plot                        #
#################################################################

### On all players ###

# It would be interesting to see if these proportion change throughout the years
#df_roles = df %>% filter(drafted==TRUE) %>% group_by(end_season)
#n <- dim(df_roles)[1]
#df_boh <- df_roles %>% count(position) %>% 
#  pivot_wider(names_from = "position", values_from = "n")
#df_boh[,-1] <- df_boh[,-1] / (df_roles %>% count())$n

#plot(df_boh$end_season, df_boh$Center)
#plot(df_boh$end_season, df_boh$Forward, lty = 2)

year = 2003
year_f =2018

par(mfrow = c(1,2))

df_position_count <- df %>% filter(start_season <= year, end_season >= year) %>% 
  count(position) %>%  pivot_wider(names_from = "position", values_from = "n")

df_position_count$Season <- year

for (i in (year+1):year_f){
  row <- df %>% filter(start_season <= i, end_season >= i) %>% 
    count(position) %>%  pivot_wider(names_from = "position", values_from = "n")
  row$Season <- i
  df_position_count <- rbind(df_position_count,row)
}

df_position_count <- df_position_count %>% mutate(Total = Center+Forward+Guard)

df_position_count

# Convert to percentage. Not really useful.
df_position_count <- df_position_count %>% mutate(Center = 100*Center/Total)  %>% 
  mutate(Forward = 100*Forward/Total) %>%
  mutate(Guard = 100*Guard/Total)

df_position_count


# Stacked 
df_temp <- df_position_count %>% pivot_longer(cols = c('Center','Forward','Guard'), 
                                              names_to = 'Position',  
                                              values_to = 'Percentage')
df_temp

# Stacked + percent
ggp1 <-ggplot(df_temp, aes(fill=Position, y=Percentage, x=Season)) +
  scale_fill_brewer(palette = "YlOrRd")+ 
  theme_minimal()+
  geom_bar(position="fill", stat="identity", show.legend = FALSE) +
  scale_x_continuous(breaks=seq(2003, 2020, 1)) +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))+
  labs(x = "Season", y = "Proportion of Roles", 
       title = "Players position - Percentage - All Players ", 
       subtitle = "Seasons from 2003 to 2018")



##########################
#   On drafted players   #
##########################



df_position_count <- df %>% filter(start_season <= year, end_season >= year, drafted == TRUE) %>% 
  count(position) %>%  pivot_wider(names_from = "position", values_from = "n")

df_position_count$Season <- year

for (i in (year+1):year_f){
  row <- df %>% filter(start_season <= i, end_season >= i, drafted == TRUE) %>% 
    count(position) %>%  pivot_wider(names_from = "position", values_from = "n")
  row$Season <- i
  df_position_count <- rbind(df_position_count,row)
}

df_position_count <- df_position_count %>% mutate(Total = Center+Forward+Guard)

df_position_count <- df_position_count %>% mutate(Center = 100*Center/Total)  %>% 
  mutate(Forward = 100*Forward/Total) %>%
  mutate(Guard = 100*Guard/Total)

df_position_count


# Stacked 
df_temp <- df_position_count %>% pivot_longer(cols = c('Center','Forward','Guard'), 
                                              names_to = 'Position',  
                                              values_to = 'Percentage')
df_temp

# Stacked + percent
ggp2 <-ggplot(df_temp, aes(fill=Position, y=Percentage, x=Season)) +
  scale_fill_brewer(palette = "YlOrRd")+ 
  theme_minimal()+
  geom_bar(position="fill", stat="identity") +
  theme(legend.title = element_blank()) +
  scale_x_continuous(breaks=seq(2003, 2018, 1)) +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))+
  labs(x = "Season", y = "Proportion of Roles", 
       title = "Players position - Percentage - Drafted ", 
       subtitle = "Seasons from 2003 to 2018")

##################################################################

library(gridExtra)

grid.arrange(ggp1, ggp2, ncol = 2)  

##################################################################




  
data

df_position_count



prova <- ggplot(df_boh, aes(x=end_season, y = Center)) +
  scale_x_continuous(breaks=seq(2003, 2020, 1)) + 
  geom_line(aes(y = Guard, colour = "var1"))+
  geom_line(aes(y = Forward, colour = "var2"))+
  scale_y_continuous(breaks=seq(0, 1, 0.1)) + ylim(c(0,1))+
  geom_line() + 
  geom_ma(ma_fun = SMA, n = 4,col="firebrick") +
  #geom_smooth(method="lm") +
  labs(x = "Season", y = "Proportion in the dataset", title = "Proportion of players per positions - Drafted players", 
       subtitle = "Seasons from 2001 to 2020")
  
prova


#######


# More than the proportion of drafted players, I would be interested in a plot 
# representing the proportion of players playing each year





# The mtcars dataset is natively available
# head(mtcars)

# A really basic boxplot.
ggplot(df, aes(x=position, y=assists)) + 
  geom_boxplot(fill="slateblue", alpha=0.6) +
  xlab('Position') + ylab('Assists')

# A really basic violinplot
ggplot(df, aes(x=position, y=height)) + 
  geom_violin(fill="slateblue", alpha=0.6, adjust = 2.3) +
  xlab('Position') + ylab('Height')

# A really basic violinplot
ggplot(df, aes(x=position, y=weight)) + 
  geom_violin(fill="slateblue", alpha=0.6, adjust = 1.5) +
  xlab('Position') + ylab('Weight')

# A really basic violinplot
ggplot(df, aes(x=position, y=drafted)) + 
  geom_violin(fill="slateblue", alpha=0.6, adjust = 1.5) +
  xlab('Position') + ylab('Weight')


df


