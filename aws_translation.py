import boto3

# TODO:
# Set regision_name
# Set Access keys in aws configuration
# Pass Src and Dest language pair from
# Following supported lanauge pair
# https://docs.aws.amazon.com/translate/latest/dg/pairs.html
def aws_translate(input_txt, src_lang, dst_lang):
  translate = boto3.client(service_name='translate', region_name='us-west-2', use_ssl=True)

  result = translate.translate_text(Text=input_txt, 
              SourceLanguageCode=src_lang, TargetLanguageCode=dst_lang)
        
  # print('TranslatedText: ' + result.get('TranslatedText'))
  return result.get('TranslatedText')

# Test translation API below
# print(aws_translate("Hello World! Testing translation", 'en', 'fr'))
# print(aws_translate("Hello World! Testing translation", 'en', 'de'))

