# Open Hours Task

This repositiory is a starter to finetuning LLMs into having a better grasp of time. This repository uses Llama3-8B as the base model, and follows the main guidelines of the contained [llama-recipes](llama-recpies-liujie-develop/README.md) repo.

## Hours

The main context of the prompts involves restaurant (weekly) opening hours. Restaurant hours are in the format below:

```
{
  "Restaurant Name": {
    "Mon": "XX:XXAM-XX:XXPM",
    "Tue": "XX:XXAM-XX:XXPM",
    "Wed": "XX:XXAM-XX:XXAM, XX:XXPM-XX:XXPM", # Breaks in opening hours
    ...
    "Sun": "", # Closed for the day
}
```

Restaurants names are generated from the OpenAI API, and hours are randomly generated. 

## Prompt Formats

Hours data are randomly pulled from a json file after generation to create prompts of varying, set formats. Integrating the sample as a dictionary, prompts would ask the following quetions:

1. Given a date and a name, return the opening hours for the named restaurant (TO_HOURS)
2. Given a date, time, and a name, return whether the named restaurant is open (TO_BOOL)
3. Given a date and a time, return whether each of the restaurants are open in the form of a dictionary (TO_LIST)

Each problem's time information can be directly given as a date and time, or the date can be given in relative terms (i.e. "tomorrow" after giving a date).

## Global Variables

The [hours_inference](hours_inference.py) script includes global variables ```USE_GPT```, which allows for OpenAI integration so long as the environmental variable is set (currently ```true```), and ```USE_DEBUG```, which displays evaluated statistics on inferences as given by [hours_debug.py](hours_debug.py).
