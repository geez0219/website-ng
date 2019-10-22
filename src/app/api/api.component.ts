import { Component, OnInit } from '@angular/core';
import { HttpClient } from '@angular/common/http'; 

@Component({
  selector: 'app-api',
  templateUrl: './api.component.html',
  styleUrls: ['./api.component.css']
})
export class ApiComponent implements OnInit {

  apiList: string[];
  currentSelection: string;
  currentAPIText: string;

  constructor(private http: HttpClient) { }

  ngOnInit() {
    // this.getAPIStructure();
    this.apiList = ['network', 'pipeline'];
    this.currentSelection = 'assets/api/network.md';
    
    this.getAPIText();

    console.log(this.currentAPIText);
  }
/*
  getAPIStructure() {
    this.http.get('assets/api/structure.json', {responseType: 'text'}).subscribe(data => {
      this.apiList = <API[]>JSON.parse(data);

      console.log(this.apiList);
    });
  }
  
  updateCurrentAPI(tutorial: API) {
    this.currentSelection = 'assets/tutorial/' + tutorial.name;

    this.getSelectedAPIText();
  }

  getSelectedAPIText() {
    this.http.get(this.currentSelection, {responseType: 'text'}).subscribe(data => {
      this.currentAPIText = data;
    });
  }
*/

  updateCurrentAPI(api: string) {
    this.currentSelection = 'assets/api/' + api + '.md';

    this.getAPIText();
  }

  getAPIText() {
    this.http.get(this.currentSelection, {responseType: 'text'}).subscribe(data => {
      this.currentAPIText = data;
    });
  }

}
