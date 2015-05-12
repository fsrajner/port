using System;
using System.Collections.Generic;
using System.Linq;
using System.ServiceModel;
using System.ServiceModel.Description;
using System.Text;
using System.Threading.Tasks;

namespace WCFServer
{
    class Program
    {
        static void Main(string[] args)
        {
            Uri baseAddress = new Uri("http://localhost:8733/Design_Time_Addresses/Calculator");

            ServiceHost host = new ServiceHost(typeof(CalculatorService), baseAddress);

            host.AddServiceEndpoint(typeof(CalculatorService), new WSHttpBinding(), "CalculatorService");

            ServiceMetadataBehavior smb = new ServiceMetadataBehavior();
            smb.HttpGetEnabled = true;
            host.Description.Behaviors.Add(smb);

            host.Open();
            Console.WriteLine("ready");
            Console.ReadKey();
        }
    }
}
